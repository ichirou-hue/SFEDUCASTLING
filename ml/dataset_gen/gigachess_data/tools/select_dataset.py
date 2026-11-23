"""Отбор примеров для датасета адаптера описания.

1) Термины: для каждого термина, которого в пуле >= --min-pool, берём до --per-term примеров (редкие первыми).
2) Добор по квотам вердиктов (?? / ? / ?! / ! / обычный).
3) «Пустые» ходы (обычный ход без термина и без причины) — не больше --max-empty от итога.
Не больше --per-game примеров из одной партии.

  python tools/select_dataset.py --inp work_v4/drafts_pool_final.jsonl work_v5/drafts_pool_final.jsonl \\
      --out work_ds/desc3000_drafts.jsonl --n 3000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcdata.io import read_jsonl, stable_unit, write_jsonl  # noqa: E402

QUOTAS = {"??": 0.15, "?": 0.15, "?!": 0.15, "!": 0.10, "обычный": 0.45}
NAG_KEY = {4: "??", 2: "?", 6: "?!", 1: "!"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--per-term", type=int, default=40)
    ap.add_argument("--min-pool", type=int, default=30, help="термины реже этого в пуле не добираем специально")
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--max-empty", type=float, default=0.15)
    ap.add_argument("--seed", default="desc-3000-v1")
    args = ap.parse_args()

    pool = [d for p in args.inp for d in read_jsonl(p)]
    pool.sort(key=lambda d: stable_unit(args.seed, d["id"]))
    terms_of = {d["id"]: set(re.findall(r"\[([^\]]+)\]", d["draft"])) for d in pool}
    empty = {d["id"] for d in pool if d["nag"] is None and not terms_of[d["id"]] and "причина:" not in d["draft"]}
    freq = Counter(t for ts in terms_of.values() for t in ts)

    chosen, ids, per_game, by_cat, got = [], set(), Counter(), Counter(), Counter()
    n_empty_max = int(args.n * args.max_empty)

    def take(d):
        if d["id"] in ids or per_game[d["group_id"]] >= args.per_game or len(chosen) >= args.n:
            return False
        if d["id"] in empty and sum(1 for c in chosen if c["id"] in empty) >= n_empty_max:
            return False
        chosen.append(d)
        ids.add(d["id"])
        per_game[d["group_id"]] += 1
        by_cat[NAG_KEY.get(d["nag"], "обычный")] += 1
        for t in terms_of[d["id"]]:
            got[t] += 1
        return True

    for t in sorted((t for t in freq if freq[t] >= args.min_pool), key=lambda t: freq[t]):
        for d in pool:
            if got[t] >= args.per_term:
                break
            if t in terms_of[d["id"]]:
                take(d)
    for cat, share in QUOTAS.items():
        target = int(args.n * share)
        for d in pool:
            if by_cat[cat] >= target:
                break
            if NAG_KEY.get(d["nag"], "обычный") == cat:
                take(d)
    for d in pool:  # добивка до n, если какой-то категории не хватило
        if len(chosen) >= args.n:
            break
        take(d)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, chosen)
    stats = {"n": len(chosen), "pool": len(pool), "by_verdict": dict(by_cat),
             "empty": sum(1 for c in chosen if c["id"] in empty), "with_intent": sum(bool(c.get("intent_used")) for c in chosen),
             "games": len(per_game), "terms": dict(got.most_common()),
             "terms_skipped_rare": sorted(t for t in freq if freq[t] < args.min_pool)}
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
