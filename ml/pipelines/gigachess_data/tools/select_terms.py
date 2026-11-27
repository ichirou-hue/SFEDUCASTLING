"""Пробная выборка с упором на заданные термины: до --per-term примеров на термин (редкие первыми),
затем случайная добивка до --n. Не больше --per-game примеров из одной партии.

  python tools/select_terms.py --inp work_v4/drafts_pool.jsonl --out work_v4/try100_drafts.jsonl --n 100
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata.io import read_jsonl, stable_unit, write_jsonl  # noqa: E402

NEW_TERMS = [
    "вскрытый шах", "двойной шах", "сквозной удар", "отвлечение", "завлечение", "рентген", "уничтожение защиты",
    "перекрытие", "освобождение линии", "промежуточный ход", "ловля фигуры", "тихий ход", "оголённый король",
    "мат по последней горизонтали", "спёртый мат", "мат Анастасии", "арабский мат", "мат «крюк»", "мат Бодена",
    "мат двумя слонами", "мат «ласточкин хвост»", "вилка", "жертва", "связка", "атака на короля",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-term", type=int, default=5)
    ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--seed", default="try-new-terms")
    ap.add_argument("--exclude", nargs="*", default=[], help="jsonl, чьи id (и партии) не брать")
    args = ap.parse_args()

    skip_ids, skip_games = set(), set()
    for p in args.exclude:
        for d in read_jsonl(p):
            skip_ids.add(d["id"])
            skip_games.add(d.get("group_id") or d["id"].rsplit("_", 1)[0])
    pool = sorted((d for d in read_jsonl(args.inp) if d["id"] not in skip_ids and d["group_id"] not in skip_games),
                  key=lambda d: stable_unit(args.seed, d["id"]))
    terms_of = {d["id"]: set(re.findall(r"\[([^\]]+)\]", d["draft"])) for d in pool}
    freq = Counter(t for ts in terms_of.values() for t in ts)
    chosen, per_game, got = [], Counter(), Counter()

    def take(d):
        if d in chosen or per_game[d["group_id"]] >= args.per_game or len(chosen) >= args.n:
            return False
        chosen.append(d)
        per_game[d["group_id"]] += 1
        for t in terms_of[d["id"]]:
            got[t] += 1
        return True

    for t in sorted(NEW_TERMS, key=lambda t: freq[t]):
        for d in pool:
            if got[t] >= args.per_term:
                break
            if t in terms_of[d["id"]]:
                take(d)
    for d in pool:
        if len(chosen) >= args.n:
            break
        take(d)
    write_jsonl(args.out, chosen)
    print(f"выбрано {len(chosen)}")
    for t in NEW_TERMS:
        print(f"  {t:30} в пуле {freq[t]:4d}  в выборке {got[t]:3d}")


if __name__ == "__main__":
    main()
