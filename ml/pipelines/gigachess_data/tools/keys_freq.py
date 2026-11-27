"""Частота тем gcdata/keys.py на наших обогащённых ходах (реальные партии, а не задачи).

Вариант: сыгранный ход + ответ движка. Если в записи есть полный вариант (engine.best_reply.pv_uci)
и предыдущий ход партии (prev_move_uci) — берём их, иначе — обрезанную линию из плана (нижняя оценка).

  python tools/keys_freq.py --inp work_v3/enriched.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata.io import read_jsonl  # noqa: E402
from gcdata.keys import line_for_record, tags  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", help="jsonl: id -> темы")
    args = ap.parse_args()

    cnt, by_bad = Counter(), Counter()
    n = n_bad = full = with_pre = 0
    out = []
    for rec in read_jsonl(args.inp):
        L, info = line_for_record(rec)
        if L is None:
            continue
        t = tags(L)
        eng = rec["facts"]["engine"]
        bad = eng["nag"] in (2, 4, 6) or bool(eng.get("allows_mate_in"))
        n += 1
        n_bad += bad
        full += info["full_pv"]
        with_pre += info["pre"]
        for x in t:
            cnt[x] += 1
            by_bad[x] += bad
        out.append({"id": rec["id"], "tags": sorted(t)})
    print(f"ходов: {n} (ошибок ?/??/мат: {n_bad}); полный вариант: {full}, предыдущий ход: {with_pre}\n")
    print(f"{'тема':22} {'ходов':>6} {'доля':>7} {'из них при ошибке':>18}")
    for t, c in cnt.most_common():
        print(f"{t:22} {c:6d} {c / n:7.1%} {by_bad[t]:18d}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
