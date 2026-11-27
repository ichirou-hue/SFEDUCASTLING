"""Шаг 8. Автоматическая перепроверка фактов и терминов (без LLM, без человека).

1. Устойчивость движка: та же позиция на большей глубине (--depth, по умолчанию 24).
   Флаг, если меняется вердикт (NAG), лучший ход при ошибке, найденные угрозы или набор терминов.
   Такие факты — «на грани порогов», им нельзя доверять как истине.
2. Согласованность термина с исходом: у термина есть ожидаемое следствие в линии движка.
   Если правило сработало, а следствия нет, — вероятная ошибка правила.
Выход: <out>.jsonl с флагами по каждому id и <out>.summary.json с долями по терминам.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcdata.concepts import TERM_PATTERNS, find_concepts
from gcdata.engine import Analyzer
from gcdata.facts import all_facts
from gcdata.io import read_jsonl, write_jsonl

_an = None


def _init(engine, depth, threat_depth, hash_mb):
    global _an
    _an = Analyzer(engine, depth=depth, threat_depth=threat_depth, hash_mb=hash_mb)


def _deep(d):
    try:
        f = all_facts(d["fen_before"], d["move_uci"])
        f["engine"] = _an.analyse_move(d["fen_before"], d["move_uci"])
        return d["id"], f, None
    except Exception:
        return d["id"], None, traceback.format_exc(limit=2)


def plan_terms(draft: str) -> set:
    return {t for t in re.findall(r"\[([^\]]+)\]", draft) if t in TERM_PATTERNS}


def net_for_mover(eng: dict, color: str, key: str = "best_reply") -> int | None:
    r = eng.get(key) or {}
    s = r.get("summary") if key == "best_reply" else eng["best_move_before"].get("line_summary")
    if not s:
        return None
    return s["net_for_white"] * (1 if color == "w" else -1)


def consistency_flags(d: dict) -> list[str]:
    """Термин сработал — проверяем его ожидаемое следствие по линии движка."""
    f = d["facts"]
    eng, dl = f["engine"], f["delta"]
    terms = plan_terms(d["draft"])
    net = net_for_mover(eng, dl["color"])
    if net is not None and dl["captured"]:
        # Баланс линии ответа не включает само взятие сходившей стороны — добавляем его.
        net += {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}[dl["captured"]["piece"].lower()]
    flags = []
    if "вилка" in terms and not eng.get("own_threat") and (net is None or net < 1):
        flags.append("вилка без угрозы и без выигрыша материала в линии")
    if "жертва" in terms and net is not None and net >= 0:
        flags.append("жертва, но в линии ответа материал не отдан")
    if "выигрыш материала" in terms and net is not None and net < 1:
        flags.append(f"выигрыш материала, но итог с учётом ответа {net:+d}")
    if "размен" in terms and net not in (0, None):
        flags.append(f"размен, но итог с учётом ответа {net:+d}")
    if "вскрытое нападение" in terms:
        tgt = next(iter(dl["discovered_attacks"]), None)
        if tgt and tgt["piece"].lower() == "p":
            flags.append("вскрытое нападение на пешку (слабый повод для термина)")
    if "упущенная возможность" in terms and eng["played_is_best"]:
        flags.append("упущенная возможность при сыгранном лучшем ходе")
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--polished", required=True, help="проверяем только принятые разборы")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--threat-depth", type=int, default=16)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ok_ids = {r["id"] for r in read_jsonl(args.polished) if r.get("answer")}
    drafts = [d for d in read_jsonl(args.drafts) if d["id"] in ok_ids]
    by_id = {d["id"]: d for d in drafts}
    print(f"к перепроверке: {len(drafts)}, глубина {args.depth}")

    results, errors = [], 0
    with Pool(args.workers, initializer=_init, initargs=(args.engine, args.depth, args.threat_depth, args.hash)) as pool:
        for n, (i, deep, err) in enumerate(pool.imap_unordered(_deep, drafts, chunksize=2), 1):
            if err:
                errors += 1
                continue
            d = by_id[i]
            e0, e1 = d["facts"]["engine"], deep["engine"]
            flags = []
            if e0["nag"] != e1["nag"]:
                flags.append(f"вердикт нестабилен: {e0['nag_glyph'] or '—'} -> {e1['nag_glyph'] or '—'}")
            if e0["nag"] in (2, 4, 6) and e0["best_move_before"]["uci"] != e1["best_move_before"]["uci"]:
                flags.append(f"лучший ход нестабилен: {e0['best_move_before']['san']} -> {e1['best_move_before']['san']}")
            t0 = plan_terms(d["draft"])
            t1 = {c.term for c in find_concepts({**deep}) if c.term in TERM_PATTERNS}
            # Сравниваем только термины, попавшие в план: остальные планировщик не использовал.
            lost = sorted(t for t in t0 if t not in t1)
            if lost:
                flags.append(f"термин не подтверждается на глубине {args.depth}: {lost}")
            flags += consistency_flags(d)
            results.append({"id": i, "terms": sorted(t0), "nag": e0["nag"], "nag_deep": e1["nag"], "flags": flags})
            if n % 100 == 0:
                print(f"  {n}")
    write_jsonl(args.out, results)

    per_term = defaultdict(lambda: [0, 0])
    kinds = Counter()
    for r in results:
        for t in r["terms"]:
            per_term[t][0] += 1
            per_term[t][1] += any(t in fl for fl in r["flags"])
        for fl in r["flags"]:
            kinds[fl.split(":")[0]] += 1
    summary = {
        "checked": len(results), "errors": errors,
        "flagged_any": sum(bool(r["flags"]) for r in results),
        "flag_kinds": kinds.most_common(),
        "per_term_flag_rate": {t: f"{b}/{a}" for t, (a, b) in sorted(per_term.items(), key=lambda x: -x[1][1] / max(1, x[1][0]))},
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
