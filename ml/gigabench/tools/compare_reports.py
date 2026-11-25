#!/usr/bin/env python3
"""Generate a self-contained comparison of two matched GigaBench reports."""
import argparse
import hashlib
import html
import json
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
main{max-width:960px;margin:auto;padding:24px 16px}h1{margin:0}
.scroll{overflow:auto;background:white;border:1px solid var(--line);border-radius:12px}
table{width:100%;min-width:540px;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:12px;text-align:left;border-bottom:1px solid var(--line)}thead{background:#eaf0f7}tfoot{background:#f0f4fa}
.up{color:#126333}.down{color:#a21b2d}.flat,.na{color:#46586a}
.up,.down,.flat,.na{font-weight:700;white-space:nowrap}small{display:block;color:#46586a}
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
        if not training and backend.get("training_performed") is False:
            training = "base"
    dataset = dataset or identity.get("dataset")
    if not base or not training:
        raise ValueError(f"Укажите базовую модель и тип обучения для {path.name}.")
    if slug(training) == "base":
        if dataset and dataset.casefold() != "none":
            raise ValueError("У базовой модели (base) датасет должен быть none.")
        dataset = "none"
    elif not dataset or dataset.casefold() == "none":
        raise ValueError(f"Укажите имя датасета для {path.name}.")
    return "--".join(map(slug, (base, training, dataset)))


def comparison_filename(before, after):
    name = f"{before}__vs__{after}.html"
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
        identity["input"] = {k: v for k, v in backend.items() if "profile" in k or "builder" in k}
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


def score_cell(earned, total):
    return f"{earned} / {total}<small>{100 * earned / total:.2f}%</small>" if total else "—"


def table_row(label, before, after):
    a, n = before
    b, m = after
    if n != m:
        raise ValueError(f"Разные знаменатели для строки {label}: {n} и {m}.")
    if not n:
        css, delta = "na", "Нет данных"
    else:
        css = "up" if b > a else "down" if b < a else "flat"
        delta = f"{100 * (b - a) / n:+.2f}".replace("-", "−") if b != a else "0,00"
        delta = delta.replace(".", ",") + " п. п."
    return (f'<tr><th scope="row">{html.escape(label)}</th><td>{score_cell(a, n)}</td>'
            f'<td>{score_cell(b, m)}</td><td class="{css}">{delta}</td></tr>')


def compare(before, after):
    a, name_a, identity_a = read_report(before)
    b, name_b, identity_b = read_report(after)
    if identity_a != identity_b:
        different = sorted(k for k in identity_a.keys() | identity_b.keys()
                           if identity_a.get(k) != identity_b.get(k))
        raise ValueError("Прогоны несопоставимы: " + ", ".join(different))
    blocks_a = {k: v["tasks"] for k, v in a["by_block"].items()}
    blocks_b = {k: v["tasks"] for k, v in b["by_block"].items()}
    if blocks_a != blocks_b:
        raise ValueError("Состав блоков различается.")
    tasks = [t for ts in blocks_a.values() for t in ts]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Задача учтена в нескольких блоках.")
    total_a, total_b = counts(a, tasks), counts(b, tasks)
    if any(total[1] != s["coverage"]["expected"] for total, s in ((total_a, a), (total_b, b))):
        raise ValueError("Сумма заданий по блокам не совпадает с покрытием.")
    rows = "".join(table_row(BLOCKS.get(block, block), counts(a, ts), counts(b, ts))
                   for block, ts in blocks_a.items())
    total = table_row("Все задания", total_a, total_b)
    return f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сравнение GigaBench</title><style>{CSS}</style>
<main><h1>Сравнение GigaBench</h1>
<p>Доля подтверждённых правильных ответов от всех заданий блока. Δ = после − до, в процентных пунктах.
Неопределённые ответы остаются в знаменателе, но не считаются доказанными ошибками.</p>
<div class="scroll" tabindex="0" role="region" aria-label="Сравнение результатов по блокам">
<table><thead><tr><th scope="col">Блок</th>
<th scope="col">До<small>{html.escape(name_a)}</small></th>
<th scope="col">После<small>{html.escape(name_b)}</small></th>
<th scope="col">Δ</th></tr></thead><tbody>{rows}</tbody><tfoot>{total}</tfoot></table></div>
</main></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="first run directory or its report.html")
    parser.add_argument("after", type=Path, help="second run directory or its report.html")
    for side in ("before", "after"):
        parser.add_argument(f"--{side}-base", help="base model ID, e.g. Qwen/Qwen3-1.7B")
        parser.add_argument(f"--{side}-training", help="training type, e.g. base, lora or full-sft")
        parser.add_argument(f"--{side}-dataset", help="your training dataset name, e.g. pilot-cells-v1")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--out-dir", type=Path, default=Path("."), help="directory for the canonical filename")
    output.add_argument("--out", type=Path, help="explicit path with the canonical filename")
    args = parser.parse_args()
    try:
        page = compare(args.before, args.after)
        before = model_slug(args.before, args.before_base, args.before_training, args.before_dataset)
        after = model_slug(args.after, args.after_base, args.after_training, args.after_dataset)
        name = comparison_filename(before, after)
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
