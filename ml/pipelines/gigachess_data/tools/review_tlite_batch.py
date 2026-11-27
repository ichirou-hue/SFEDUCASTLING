"""Разбор последней пачки T-lite: статистика новых строк, причины повторов и несколько примеров «план → текст».

  python tools/review_tlite_batch.py --since 343 --show 6
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--since", type=int, required=True, help="номер строки simple_polished.jsonl, с которой начинается пачка")
ap.add_argument("--show", type=int, default=6)
ap.add_argument("--seed", default="review")
args = ap.parse_args()

rows = [json.loads(l) for l in open("work_tlite/simple_polished.jsonl", encoding="utf-8")]
new = rows[args.since:]
ids = {r["id"] for r in new}
rej = [json.loads(l) for l in open("work_tlite/simple_polished.rejected.jsonl", encoding="utf-8")]
rej = [r for r in rej if r["id"] in ids]
drafts = {json.loads(l)["id"]: json.loads(l) for l in open("work_tlite/simple_drafts.jsonl", encoding="utf-8")}
ok = [r for r in new if r.get("answer")]
print(f"в пачке {len(new)}: принято {len(ok)}, отклонено {len(new) - len(ok)}; "
      f"попыток {dict(Counter(r.get('attempts') for r in ok))}; всего строк {len(rows)}")
print("причины повторов:", Counter(e.split(":")[0][:60] for r in rej for e in r["reasons"]).most_common(8))
random.Random(args.seed + str(args.since)).shuffle(ok)
for r in ok[: args.show]:
    print("\nПЛАН:", drafts[r["id"]]["draft"].replace("\n", " | ")[:450])
    print("ТЕКСТ:", r["answer"])
