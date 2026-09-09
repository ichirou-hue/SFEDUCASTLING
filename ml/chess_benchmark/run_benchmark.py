import argparse
import glob
import importlib
import inspect
import json
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table

from logger import BenchmarkLogger
from utils import init_model_runner

console = Console()
MODULE_MAX_SCORE = 500.0


def load_test_classes(filter_pattern: str = None):
    test_files = sorted(glob.glob("tests/[0-9]*.py"))
    test_classes = []
    for path in test_files:
        module_name = os.path.splitext(os.path.basename(path))[0]
        if filter_pattern and filter_pattern not in module_name:
            continue
        mod = importlib.import_module(f"tests.{module_name}")
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and hasattr(attr, "name") and hasattr(attr, "evaluate"):
                test_classes.append(attr)
                break
    return test_classes


def main():
    parser = argparse.ArgumentParser(description="In-Memory Modular Chess Benchmark")
    parser.add_argument("--model-path", type=str, required=True, help="Путь к модели или API URL")
    parser.add_argument("--base-model-path", type=str, default=None, help="Базовая модель для LoRA")
    parser.add_argument("--test", type=str, default=None, help="Фильтр по номеру теста (01, 02...)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ru"], help="Язык (en / ru)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод хода тестирования")
    parser.add_argument("--show-answers", action="store_true", help="Вывести сводную таблицу всех 33 ответов в терминал")
    parser.add_argument("--no-wandb", action="store_true", help="Отключить отправку в W&B")
    parser.add_argument("--wandb-project", type=str, default="dani4_10-d/chess-coach-benchmark")
    args = parser.parse_args()

    model_name = "GigaChess-Cloud-API" if args.model_path.startswith("http") else os.path.basename(os.path.normpath(args.model_path))

    test_classes = load_test_classes(args.test)
    if not test_classes:
        console.print(f"[bold red]Ошибка:[/] Тесты не найдены по фильтру: {args.test}")
        return

    runner = init_model_runner(args.model_path, args.base_model_path)

    category_rows = []
    detailed_rows = []
    total_correct = 0
    total_samples = 0
    total_score = 0.0

    for test_cls in test_classes:
        test_obj = test_cls()
        if not os.path.exists(test_obj.dataset_path):
            continue

        with open(test_obj.dataset_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f]

        passed = 0
        console.print(f"\n[bold yellow]▶ Запуск {test_obj.name}[/] ({len(items)} задач)...")

        for item in items:
            fen = item.get("fen", "")
            expected = item.get("expected", item.get("expected_san", ""))

            if hasattr(test_obj, "format_prompt"):
                sig = inspect.signature(test_obj.format_prompt)
                prompt = test_obj.format_prompt(item, lang=args.lang) if "lang" in sig.parameters else test_obj.format_prompt(item)
            else:
                prompt = item.get("question_ru" if args.lang == "ru" and "question_ru" in item else "question", "")

            pred = runner.generate(prompt=prompt, fen=fen)

            eval_sig = inspect.signature(test_obj.evaluate)
            eval_kwargs = {}
            if "item" in eval_sig.parameters:
                eval_kwargs["item"] = item
            if "lang" in eval_sig.parameters:
                eval_kwargs["lang"] = args.lang

            is_ok, reason = test_obj.evaluate(pred, expected, **eval_kwargs)
            if is_ok:
                passed += 1

            status_str = "✔ OK" if is_ok else "✘ FAIL"
            q_text = item.get("question_ru" if args.lang == "ru" and "question_ru" in item else "question_en", item.get("question", ""))

            detailed_rows.append([
                test_obj.name,
                status_str,
                q_text,
                fen,
                str(expected),
                str(pred),
                reason
            ])

            if args.verbose:
                status_color = "[bold green]✔ OK[/]" if is_ok else "[bold red]✘ FAIL[/]"
                console.print(f"  {status_color} [{item.get('id', '')}] {q_text}")
                console.print(f"     [dim]Ожидалось:[/] [white]{expected}[/]")
                console.print(f"     [dim]Ответ:[/]     [yellow]\"{pred}\"[/]")
                if not is_ok:
                    console.print(f"     [bold red]Причина:[/]   [red]{reason}[/]")
                console.print()

        acc = (passed / len(items)) * 100 if items else 0
        module_score = (passed / len(items)) * MODULE_MAX_SCORE if items else 0
        total_score += module_score

        category_rows.append([
            test_obj.name,
            test_obj.dataset_path,
            f"{passed} / {len(items)}",
            round(module_score, 1),
            round(acc, 1)
        ])

        total_correct += passed
        total_samples += len(items)

    max_possible_score = len(category_rows) * MODULE_MAX_SCORE
    overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    # 1. Печать итоговой таблицы по категориям
    results_table = Table(
        title=f"\n♟️  ИТОГИ БЕНЧМАРКА: [bold yellow]{model_name}[/] (Язык: [cyan]{args.lang.upper()}[/])",
        show_lines=True
    )
    results_table.add_column("Модуль", style="cyan", width=24)
    results_table.add_column("Датасет", style="dim", width=32)
    results_table.add_column("Верно", justify="center", width=10)
    results_table.add_column("Score", justify="center", style="bold magenta", width=14)
    results_table.add_column("Accuracy", justify="center", style="bold green", width=12)

    for row in category_rows:
        results_table.add_row(row[0], row[1], row[2], f"{row[3]:.1f} / {MODULE_MAX_SCORE:.0f}", f"{row[4]:.1f}%")

    console.print(results_table)
    console.print(
        f"\n[bold yellow]ИТОГОВЫЙ СЧЕТ:[/] [bold green]{total_score:.1f} / {max_possible_score:.0f}[/] "
        f"([bold cyan]Accuracy: {overall_acc:.1f}%[/], {total_correct}/{total_samples} задач)\n"
    )

    # 2. Опциональный детальный вывод всех ответов в консоль
    if args.show_answers:
        details_table = Table(title="📋 ПОЛНЫЙ ЛОГ ОТВЕТОВ МОДЕЛИ", show_lines=True)
        details_table.add_column("Модуль", style="cyan", width=18)
        details_table.add_column("Статус", justify="center", width=8)
        details_table.add_column("Вопрос", style="white", width=35)
        details_table.add_column("Ожидание", style="green", width=16)
        details_table.add_column("Ответ модели", style="yellow", width=24)
        details_table.add_column("Причина / Ошибка", style="red", width=28)

        for row in detailed_rows:
            st_color = "[bold green]✔ OK[/]" if "OK" in row[1] else "[bold red]✘ FAIL[/]"
            details_table.add_row(row[0], st_color, row[2], row[4], row[5], row[6])

        console.print(details_table)
        console.print()

    # 3. Отправка в W&B напрямую из RAM
    if not args.no_wandb:
        summary_metrics = {
            "Total_Score": round(total_score, 1),
            "Max_Score": max_possible_score,
            "Overall_Accuracy": round(overall_acc, 1),
            "Solved_Tasks": total_correct,
            "Total_Tasks": total_samples,
        }
        for row in category_rows:
            summary_metrics[f"Score/{row[0]}"] = row[3]
            summary_metrics[f"Accuracy/{row[0]}"] = row[4]

        run_name = f"{model_name}_{args.lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {
            "model": model_name,
            "lang": args.lang,
            "target": args.model_path,
            "filter": args.test or "ALL"
        }
        logger = BenchmarkLogger(project=args.wandb_project, run_name=run_name, config=config)
        logger.log_results(summary_metrics, category_rows, detailed_rows)


if __name__ == "__main__":
    main()