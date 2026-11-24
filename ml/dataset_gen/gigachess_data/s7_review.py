"""Шаг 7. Таблица для ручной проверки шахматистом.

random — --n случайных разборов из основной выборки (фиксированный seed: выборка зафиксирована
         до проверки, подбирать «удобные» примеры нельзя);
term:X — все дополнительные примеры по терминам (проверка правил словаря).
CSV в UTF-8 с BOM и разделителем «;» — открывается в Excel без настройки кодировки.
Колонки чек-листа заполняет проверяющий: да / нет, комментарий — при «нет».
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcdata.io import read_jsonl, stable_unit

CHECKLIST = [
    "Факты верны (ход, оценка, линии)",
    "Термин уместен (шахматист назвал бы так же)",
    "Текст точно передаёт план",
    "Читается как разбор тренера",
    "Комментарий",
]


def rows(drafts_path: str, polished_path: str, label) -> list[dict]:
    answers = {r["id"]: r.get("answer") for r in read_jsonl(polished_path)}
    out = []
    for d in read_jsonl(drafts_path):
        if not answers.get(d["id"]):
            continue
        out.append({
            "набор": label(d),
            "id": d["id"],
            "вердикт движка": d["verdict"] + (f" ({d['cpl']} cp)" if d["cpl"] else ""),
            "термины": ", ".join(re.findall(r"\[([^\]]+)\]", d["draft"])),
            "позиция (lichess)": "https://lichess.org/analysis/" + d["fen_before"].replace(" ", "_"),
            "вопрос (вход модели)": d["question"],
            "ответ (таргет)": answers[d["id"]],
            "план из скриптов": d["draft"],
            "замысел из партии": (d.get("intent") or {}).get("text") or "",
            **{c: "" for c in CHECKLIST},
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-drafts", required=True)
    ap.add_argument("--main-polished", required=True)
    ap.add_argument("--terms-drafts")
    ap.add_argument("--terms-polished")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per-term", type=int, default=25)
    ap.add_argument("--seed", default="review-300")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    main_rows = rows(args.main_drafts, args.main_polished, lambda d: "random")
    main_rows = sorted(main_rows, key=lambda r: stable_unit(args.seed, r["id"]))[: args.n]
    term_rows = []
    if args.terms_drafts:
        term_rows = rows(args.terms_drafts, args.terms_polished, lambda d: f"term:{d['probe_term']}")
        term_rows.sort(key=lambda r: (r["набор"], r["id"]))
    # Редкие термины добираем до --per-term уже готовыми разборами основной выборки (не из random).
    have = Counter(t for r in main_rows + term_rows for t in r["термины"].split(", ") if t)
    used = {r["id"] for r in main_rows + term_rows}
    for r in rows(args.main_drafts, args.main_polished, lambda d: "main"):
        need = [t for t in r["термины"].split(", ") if t and have[t] < args.per_term]
        if need and r["id"] not in used:
            r["набор"] = f"term:{need[0]} (из основной)"
            term_rows.append(r)
            used.add(r["id"])
            have.update(t for t in r["термины"].split(", ") if t)
    all_rows = main_rows + term_rows
    for i, r in enumerate(all_rows, 1):
        r["№"] = i
    cols = ["№"] + [k for k in all_rows[0] if k != "№"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=";")
        w.writeheader()
        w.writerows(all_rows)
    print(f"строк: {len(all_rows)} (random: {len(main_rows)}, по терминам: {len(term_rows)}) -> {args.out}")


if __name__ == "__main__":
    main()
