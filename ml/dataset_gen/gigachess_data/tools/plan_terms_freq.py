"""Частота терминов в планах (после условий concepts.py) на обогащённых ходах + примеры.

  python tools/plan_terms_freq.py --inp work_v4/enriched.jsonl --max-plies 6 --examples вилка,жертва
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata import keys  # noqa: E402
from gcdata.draft import build_draft  # noqa: E402
from gcdata.io import read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--max-plies", type=int, default=12, help="сколько полуходов ответа видят ключи")
    ap.add_argument("--examples", default="", help="термины через запятую: показать по 2 плана")
    args = ap.parse_args()

    orig = keys._reply_pv
    keys._reply_pv = lambda facts, max_plies: orig(facts, min(max_plies, args.max_plies))
    want = [t for t in args.examples.split(",") if t]
    cnt, ex = Counter(), {t: [] for t in want}
    n = 0
    for rec in read_jsonl(args.inp):
        n += 1
        d = build_draft(rec, None)
        for t in set(re.findall(r"\[([^\]]+)\]", d)):
            cnt[t] += 1
            if t in ex and len(ex[t]) < 2:
                ex[t].append((rec["id"], d))
    print(f"ходов: {n}, вариант для ключей: до {args.max_plies} полуходов")
    for t, k in cnt.most_common():
        print(f"{t:30} {k:5d} {k / n:6.1%}")
    for t in want:
        for i, d in ex[t]:
            print(f"\n== {t} {i}\n{d}")


if __name__ == "__main__":
    main()
