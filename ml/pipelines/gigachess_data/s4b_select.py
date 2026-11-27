"""Шаг 4b. Отбор пилотной выборки для проверки на ошибки.

main  — n планов, стратифицированных по вердикту (у ошибок отдельные квоты, иначе их почти
        не будет) и по фазе партии, не больше --per-game ходов из одной партии;
terms — дополнительно до --per-term примеров на каждый термин словаря, не пересекаясь с main:
        редкие правила в случайную выборку почти не попадают, а проверить их нужно.
Отбор детерминирован (--seed), чтобы выборку можно было зафиксировать до проверки.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess

from gcdata.concepts import TERM_PATTERNS
from gcdata.io import read_jsonl, stable_unit, write_jsonl

QUOTAS = {"??": 0.15, "?": 0.15, "?!": 0.15, "!": 0.10, "обычный": 0.45}


def verdict_bucket(d: dict) -> str:
    return {4: "??", 2: "?", 6: "?!", 1: "!"}.get(d["nag"], "обычный")


def phase(d: dict) -> str:
    pieces = len(chess.Board(d["fen_before"]).piece_map())
    if pieces <= 12:
        return "эндшпиль"
    if (d.get("ply") or 99) <= 20:
        return "дебют"
    return "миттельшпиль"


def terms_of(d: dict) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", d["draft"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, help="drafts.jsonl пула")
    ap.add_argument("--out-main", required=True)
    ap.add_argument("--out-terms", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--per-term", type=int, default=25)
    ap.add_argument("--seed", default="pilot-1000")
    args = ap.parse_args()

    pool = sorted(read_jsonl(args.inp), key=lambda d: stable_unit(args.seed, d["id"]))
    by_bucket = defaultdict(lambda: defaultdict(list))
    for d in pool:
        by_bucket[verdict_bucket(d)][phase(d)].append(d)

    per_game = Counter()
    chosen, chosen_ids = [], set()

    def take(bucket: str, k: int) -> int:
        """Берём k из корзины, чередуя фазы партии; соблюдаем лимит на партию."""
        phases = by_bucket[bucket]
        got = 0
        while got < k and any(phases.values()):
            for ph in sorted(phases):
                if got >= k:
                    break
                while phases[ph]:
                    d = phases[ph].pop(0)
                    if per_game[d["group_id"]] < args.per_game and d["id"] not in chosen_ids:
                        per_game[d["group_id"]] += 1
                        chosen.append(d)
                        chosen_ids.add(d["id"])
                        got += 1
                        break
        return got

    short = 0
    for b, q in QUOTAS.items():
        if b == "обычный":
            continue
        want = round(args.n * q)
        short += want - take(b, want)
    take("обычный", args.n - len(chosen))  # недобор по редким вердиктам добираем обычными ходами
    for b in QUOTAS:  # если и обычных не хватило — любыми
        if len(chosen) < args.n:
            take(b, args.n - len(chosen))

    extra, per_term = [], Counter()
    for d in pool:
        if d["id"] in chosen_ids:
            continue
        need = [t for t in terms_of(d) if t in TERM_PATTERNS and per_term[t] < args.per_term]
        if need:
            d = {**d, "probe_term": need[0]}
            for t in terms_of(d):
                per_term[t] += 1
            extra.append(d)

    write_jsonl(args.out_main, chosen)
    write_jsonl(args.out_terms, extra)
    term_main = Counter(t for d in chosen for t in terms_of(d))
    stats = {
        "main": len(chosen),
        "games_in_main": len({d["group_id"] for d in chosen}),
        "verdicts": Counter(verdict_bucket(d) for d in chosen),
        "phases": Counter(phase(d) for d in chosen),
        "intent_in_input": sum(d["intent_used"] for d in chosen),
        "rare_verdict_shortfall": short,
        "terms_extra": len(extra),
        "terms_total_main_plus_extra": {t: term_main[t] + per_term[t] for t in TERM_PATTERNS},
    }
    Path(args.out_main).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
