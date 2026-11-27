"""Убирает из polished примеры, у которых перепроверка движком (s8) нашла нестабильный вердикт
или неподтверждённый термин: таким фактам нельзя доверять как истине.

  python tools/drop_unstable.py --polished P.jsonl --autocheck A.jsonl --out P_clean.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata.io import read_jsonl, write_jsonl  # noqa: E402

DROP = ("вердикт нестабилен", "лучший ход нестабилен", "термин не подтверждается")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polished", required=True)
    ap.add_argument("--autocheck", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bad, why = set(), Counter()
    for r in read_jsonl(args.autocheck):
        hits = [f for f in r["flags"] if f.startswith(DROP)]
        if hits:
            bad.add(r["id"])
            why.update(h.split(":")[0] for h in hits)
    rows = [r for r in read_jsonl(args.polished) if r.get("answer") and r["id"] not in bad]
    write_jsonl(args.out, rows)
    print(f"оставлено {len(rows)}, убрано {len(bad)}: {dict(why)}")


if __name__ == "__main__":
    main()
