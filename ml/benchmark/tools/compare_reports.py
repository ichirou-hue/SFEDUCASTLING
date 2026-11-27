#!/usr/bin/env python3
"""Generate a self-contained comparison of matched GigaBench reports."""
import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path

BLOCKS = {
    "board_reading": "Чтение доски", "rules": "Правила", "tactics": "Тактика",
    "move_choice": "Выбор хода", "commentary": "Комментирование и оценка позиции",
    "diagnostics": "Ручные крайние случаи",
}

CSS = """
:root{color-scheme:light;--ink:#172532;--bg:#f3f6f9;--line:#dbe3eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{width:100%;margin:0;padding:16px}h1{margin:0}
.scroll{width:100%;overflow:auto;background:white;border:1px solid var(--line);border-radius:12px}
table{width:100%;min-width:540px;table-layout:fixed;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;text-align:left;vertical-align:top;overflow-wrap:anywhere;border-bottom:1px solid var(--line)}
thead th:not(:first-child),td{text-align:center}
thead{background:#eaf0f7}tfoot{background:#f0f4fa}
.up{color:#126333}.down{color:#a21b2d}.flat,.na{color:#46586a}
.up,.down,.flat,.na{font-weight:700;white-space:nowrap}small{display:block;color:#46586a}
.section th{background:#eaf0f7;color:var(--ink)}
"""


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_dir(path):
    return path.parent if path.name in {"report.html", "summary.json"} and path.is_file() else path


def slug(value):
    value = re.sub(r"[^\w.]+|_+", "-", value.casefold()).strip(".-")
    if not value:
        raise ValueError("Компонент имени модели не содержит букв или цифр.")
    return value


def model_slug(path, base=None, training=None, dataset=None):
    path = report_dir(path)
    run = load(path / "run.json")
    backend = run["backend"]
    identity = run.get("model_identity", {})
    base = base or identity.get("base_model")
    training = training or identity.get("training_type")
    paired = not (path / "manifest.json").exists() and path.name in {"base", "lora"}
    if paired:
        settings = load(path.parent / "settings.json")
        base = base or settings["model"]
        training = training or path.name
    else:
        base = base or backend.get("model")
        if not training:
            if backend.get("adapter"):
                training = "lora"
            elif backend.get("training_performed") is False or path.name == "base":
                training = "base"
    dataset = dataset or identity.get("dataset")
    if not dataset and training and slug(training) != "base" and path.name.startswith("lora-"):
        dataset = path.name.removeprefix("lora-")
    if not base or not training:
        raise ValueError(f"Укажите базовую модель и тип обучения для {path.name}.")
    if slug(training) == "base":
        if dataset and dataset.casefold() != "none":
            raise ValueError("У базовой модели (base) датасет должен быть none.")
        dataset = "none"
    elif not dataset or dataset.casefold() == "none":
        raise ValueError(f"Укажите имя датасета для {path.name}.")
    return "--".join(map(slug, (base, training, dataset)))


def comparison_filename(*models):
    if len(models) < 2:
        raise ValueError("Для сравнения нужны базовый прогон и хотя бы один другой.")
    name = "__vs__".join(models) + ".html"
    if len(name.encode("utf-8")) > 255:
        raise ValueError("Имя файла слишком длинное: задайте более короткие имена модели или датасета.")
    return name


def verify_pair_files(root, member):
    hashes = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        checksum, name = line.split(maxsplit=1)
        hashes[name.strip()] = checksum
    for name in ("settings.json", "queries.jsonl", "engine/engine.json",
                 *(f"{member}/{f}" for f in ("cases.jsonl", "summary.json", "run.json"))):
        if hashes.get(name) != digest(root / name):
            raise ValueError(f"Хеш {name} не совпадает с SHA256SUMS.")


def read_report(path):
    path = report_dir(path)
    summary, run = load(path / "summary.json"), load(path / "run.json")
    coverage = summary["coverage"]
    if summary.get("is_mock") or not coverage["complete"] or coverage["completed"] != coverage["expected"]:
        raise ValueError("Нужны полные прогоны модели, не mock.")
    settings_path = path.parent / "settings.json"
    if not (path / "manifest.json").exists() and settings_path.is_file():
        if path.name not in {"base", "lora"}:
            raise ValueError("Ожидался каталог base или lora внутри парного прогона.")
        verify_pair_files(path.parent, path.name)
        settings = load(settings_path)
        identity = {
            "cases": digest(path / "cases.jsonl"), "queries": digest(path.parent / "queries.jsonl"),
            "input": settings["input"], "generation": {
                key: settings[key] for key in ("engine", "temperature", "thinking", "max_new_tokens")
            }, "stockfish": settings.get("stockfish"),
        }
        if identity["cases"] != settings["source_cases_sha256"]:
            raise ValueError("Задания не совпадают с settings.json.")
        if identity["stockfish"] != load(path.parent / "engine/engine.json"):
            raise ValueError("Настройки Stockfish не совпадают с settings.json.")
        # This layout shares the rendered prompts, scorer and engine across both variants.
        identity["pair"] = path.parent.resolve()
    else:
        manifest_path = path / "manifest.json"
        manifest = load(manifest_path)
        for name, file in (("manifest", manifest_path), ("responses", path / "responses.jsonl")):
            expected = digest(file)
            if any(meta.get(f"{name}_sha256") != expected for meta in (run, summary)):
                raise ValueError(f"{file.name} не совпадает с хешами в run.json и summary.json.")
        identity = {k: manifest[k] for k in (
            "cases_sha256", "queries_sha256", "code_sha256", "config", "chess_module_sha256",
        )}
        for name in ("cases", "queries"):
            if digest(path / f"{name}.jsonl") != identity[f"{name}_sha256"]:
                raise ValueError(f"Хеш {name}.jsonl не совпадает с manifest.json.")
        backend = run["backend"]
        identity["input"] = {
            k: v for k, v in backend.items()
            if k != "model_profile" and (
                "profile" in k or "builder" in k or k in {"template", "numeric_inputs_sha256"}
            )
        }
        identity["generation"] = {k: backend.get(k) for k in ("thinking", "do_sample", "engine")}
        engine = path / "engine/engine.json"
        identity["stockfish"] = load(engine) if engine.exists() else None
        identity["rescoring"] = summary.get("rescored", {}).get("scoring_code_sha256")
    return summary, run["backend"].get("model") or path.name, identity


def counts(summary, tasks):
    rows = [summary["by_task"][task] for task in tasks]
    if any(not 0 <= r["correct"] <= r["records"] for r in rows):
        raise ValueError("Недопустимые числа правильных ответов или заданий.")
    return sum(r["correct"] for r in rows), sum(r["records"] for r in rows)


def score_cell(earned, total, delta=None):
    if not total:
        return "—" if delta is None else f"—<small>{delta}</small>"
    percentage = f"{100 * earned / total:.2f}%"
    if delta is not None:
        percentage += f" {delta}"
    return f"{earned} / {total}<small>{percentage}</small>"


def delta_markup(baseline, value, lower_is_better=False):
    if baseline is None or value is None:
        return '<span class="na">(Δ Нет данных)</span>'
    delta = value - baseline
    if math.isclose(delta, 0, abs_tol=1e-12):
        css, text = "flat", "0,00"
    else:
        better = delta < 0 if lower_is_better else delta > 0
        css = "up" if better else "down"
        text = f"{delta:+.2f}".replace("-", "−").replace(".", ",")
    return f'<span class="{css}">(Δ {text} п. п.)</span>'


def table_row(label, values, lower_is_better=False):
    totals = {total for _, total in values}
    if len(totals) != 1:
        raise ValueError(f"Разные знаменатели для строки {label}: {sorted(totals)}.")
    total = values[0][1]
    baseline = 100 * values[0][0] / total if total else None
    cells = [f"<td>{score_cell(values[0][0], total)}</td>"]
    for earned, count in values[1:]:
        value = 100 * earned / count if count else None
        delta = delta_markup(baseline, value, lower_is_better)
        cells.append(f"<td>{score_cell(earned, count, delta)}</td>")
    return f'<tr><th scope="row">{html.escape(label)}</th>{"".join(cells)}</tr>'


def metric_cell(value, sample_size, delta=None):
    percentage = f"{value:.2f}%"
    if delta is not None:
        percentage += f" {delta}"
    return f"{percentage}<small>n={sample_size}</small>"


def metric_row(label, values, sample_size, lower_is_better=False):
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError(f"Недопустимое значение критерия {label}.")
    cells = [f"<td>{metric_cell(values[0], sample_size)}</td>"]
    baseline = values[0]
    for value in values[1:]:
        delta = delta_markup(baseline, value, lower_is_better)
        cells.append(f"<td>{metric_cell(value, sample_size, delta)}</td>")
    return f'<tr><th scope="row">{html.escape(label)}</th>{"".join(cells)}</tr>'


def table_header(labels):
    cells = ['<th scope="col">Показатель</th>',
             f'<th scope="col">База<small>{html.escape(labels[0])}</small></th>']
    for label in labels[1:]:
        escaped = html.escape(label)
        cells.append(f'<th scope="col">Модель<small>{escaped}</small></th>')
    return f"<tr>{''.join(cells)}</tr>"


def compare(*runs, labels=None):
    if len(runs) < 2:
        raise ValueError("Для сравнения нужны базовый прогон и хотя бы один другой.")
    reports = [read_report(path) for path in runs]
    summaries = [report[0] for report in reports]
    identities = [report[2] for report in reports]
    baseline_identity = identities[0]
    for identity in identities[1:]:
        if identity != baseline_identity:
            different = sorted(k for k in baseline_identity.keys() | identity.keys()
                               if baseline_identity.get(k) != identity.get(k))
            raise ValueError("Прогоны несопоставимы: " + ", ".join(different))
    if labels is None:
        labels = [report[1] for report in reports]
        if len(set(labels)) != len(labels):
            labels = [f"{name} ({report_dir(path).name})"
                      for name, path in zip(labels, runs, strict=True)]
    if len(labels) != len(reports) or len(set(labels)) != len(labels):
        raise ValueError("Идентификаторы прогонов должны быть уникальными.")

    block_tasks = {
        block: data["tasks"] for block, data in summaries[0]["by_block"].items()
    }
    for summary in summaries[1:]:
        if {k: v["tasks"] for k, v in summary["by_block"].items()} != block_tasks:
            raise ValueError("Состав блоков различается.")
    tasks = [task for task_list in block_tasks.values() for task in task_list]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Задача учтена в нескольких блоках.")
    totals = [counts(summary, tasks) for summary in summaries]
    if any(total[1] != summary["coverage"]["expected"]
           for total, summary in zip(totals, summaries, strict=True)):
        raise ValueError("Сумма заданий по блокам не совпадает с покрытием.")
    rows = "".join(
        table_row(
            BLOCKS.get(block, block),
            [counts(summary, block_tasks[block]) for summary in summaries],
        )
        for block in block_tasks
    )
    total = table_row("Все задания", totals)

    commentary = [summary.get("commentary_auto_metrics") for summary in summaries]
    if any(metrics is not None for metrics in commentary):
        if any(metrics is None for metrics in commentary):
            raise ValueError("Не во всех прогонах есть критерии M4-auto–M7-auto.")
        expected_metrics = {
            (metrics.get("criteria_version"), metrics.get("n"), metrics.get("expected"))
            for metrics in commentary
        }
        if len(expected_metrics) != 1 or any(
            not metrics.get("complete") or metrics.get("n") != metrics.get("expected")
            for metrics in commentary
        ):
            raise ValueError("Критерии M4-auto–M7-auto должны быть полными и одинаковыми.")
        sample_size = commentary[0]["n"]
        criteria = (
            ("M4-auto — проверяемые утверждения", "M4", False),
            ("M5-auto — раскрытие идей", "M5", False),
            ("M6-auto — критические ошибки (меньше — лучше)", "M6", True),
            ("M7-auto — структура объяснения", "M7", False),
            ("Принятые объяснения", "accepted_percent", False),
        )
        auto_rows = "".join(
            metric_row(
                label,
                [float(metrics[key]) for metrics in commentary],
                sample_size,
                lower_is_better,
            )
            for label, key, lower_is_better in criteria
        )
        auto_section = (
            f'<tr class="section"><th colspan="{1 + len(reports)}">'
            "M4-auto–M7-auto (проценты)</th></tr>" + auto_rows
        )
    else:
        auto_section = ""
    min_width = max(1200, 360 + 240 * len(reports))
    column_width = 100 / (len(reports) + 1)
    column_group = (
        "<colgroup>"
        + "".join(f'<col style="width:{column_width:.4f}%">' for _ in range(len(reports) + 1))
        + "</colgroup>"
    )
    return f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сравнение GigaBench</title><style>{CSS}</style>
<main><h1>Сравнение GigaBench</h1>
<p>Прогонов в отчёте: {len(reports)}. Первый — база; каждая Δ рассчитана относительно него в процентных пунктах.
В блоках показана доля подтверждённо правильных ответов от всех заданий.
Неопределённые ответы остаются в знаменателе, но не считаются доказанными ошибками.</p>
<p>В M4-auto–M7-auto показаны автоматические критерии, а не экспертные оценки;
для M6-auto меньшее значение лучше.</p>
<div class="scroll" tabindex="0" role="region" aria-label="Сравнение результатов по блокам">
<table style="min-width:{min_width}px">{column_group}<thead>{table_header(labels)}</thead>
<tbody><tr class="section"><th colspan="{1 + len(reports)}">Блоки бенчмарка</th></tr>
{rows}{auto_section}</tbody><tfoot>{total}</tfoot></table></div>
</main></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+",
                        help="run directories/reports; first is the baseline")
    for side in ("before", "after"):
        parser.add_argument(f"--{side}-base", help="base model ID, e.g. Qwen/Qwen3-1.7B")
        parser.add_argument(f"--{side}-training", help="training type, e.g. base, lora or full-sft")
        parser.add_argument(f"--{side}-dataset", help="your training dataset name, e.g. pilot-cells-v1")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out-dir", type=Path, default=Path("."), help="directory for the canonical filename")
    output.add_argument("--out", type=Path, help="explicit path with the canonical filename")
    args = parser.parse_args()
    try:
        if len(args.runs) < 2:
            parser.error("Передайте базовый прогон и хотя бы один сравниваемый.")
        overrides = (
            (args.before_base, args.before_training, args.before_dataset),
            (args.after_base, args.after_training, args.after_dataset),
        )
        labels = [
            model_slug(
                path,
                *(overrides[index] if index < len(overrides) else (None, None, None)),
            )
            for index, path in enumerate(args.runs)
        ]
        page = compare(*args.runs, labels=labels)
        name = comparison_filename(*labels)
        path = args.out or args.out_dir / name
        if path.name != name:
            raise ValueError(f"Ожидаемое имя файла: {name}. Используйте --out-dir для выбора каталога.")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(page)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(path)  # noqa: T201 -- CLI output path


if __name__ == "__main__":
    main()
