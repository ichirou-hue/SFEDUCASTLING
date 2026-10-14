"""Добирает в backend/knowledge/puzzles.json НЕ-матовые задачи по диапазонам рейтинга.

Задача скрипта — снизить долю матов на низких рейтингах. В дампе Lichess задачи
до 800 рейтинга почти сплошь маты (500-600: ~91%, 600-700: ~80%), а конструктор
build_puzzles.py дополнительно ранжирует маты первыми (у них first_adv = 100000,
максимум). В результате на рейтинге ~600 выдача состояла из одних матов в 1-2 хода.

Для каждого диапазона скрипт считает текущую долю матов, вычисляет, сколько
не-матовых задач нужно добавить, чтобы доля упала до цели, и добирает их из
дампа с ротацией по темам (вилки, связки, сквозные удары, эндшпиль и т.д.).
Уже отобранные пазлы не пересобираются — только дедупликация по PuzzleId.

Запуск:
    python -m backend.knowledge.build_balanced_puzzles

Источник — официальная база database.lichess.org (тот же формат, что и в
build_puzzles.py: FEN до хода соперника, Moves содержит ход соперника первым).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import time

from backend.knowledge.build_puzzles import (
    _cheap_valid,
    _decode,
    _primary_theme,
    _validate,
)

DEFAULT_SOURCE = r"C:\Users\ASUS\AppData\Local\Temp\opencode\lichess_db_puzzle.csv.zst"
DEFAULT_SF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "stockfish_engine", "stockfish.exe")
)

MATE_THEMES = {
    "mate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
    "smotheredMate",
    "backRankMate",
}

ENDGAME_THEMES = {
    "endgame",
    "rookEndgame",
    "pawnEndgame",
    "queenEndgame",
    "knightEndgame",
    "bishopEndgame",
    "opposition",
    "passedPawn",
}

# lo:hi:целевая_доля_матов:мин_добавок_для_разнообразия_тем
DEFAULT_BANDS = (
    "500:600:0.30:0,"
    "600:700:0.30:0,"
    "700:800:0.28:0,"
    "800:900:0.25:0,"
    "900:1100:0.22:50,"
    "1100:1300:0.20:30"
)


def _is_mate(themes: list[str]) -> bool:
    return bool(set(themes) & MATE_THEMES)


def _group_of(puzzle: dict, metrics: dict) -> str:
    if metrics["solver_mates"]:
        return "mate"
    theme = _primary_theme(puzzle)
    if theme == "other" and set(puzzle.get("themes", [])) & ENDGAME_THEMES:
        return "endgame"
    return theme


def _parse_bands(spec: str) -> list[tuple[int, int, float, int]]:
    bands = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, hi_s, target_s, min_s = part.split(":")
        bands.append((int(lo_s), int(hi_s), float(target_s), int(min_s)))
    if not bands:
        raise SystemExit("--bands пуст")
    return bands


def _zstd_reader(fh):
    import zstandard as zstd

    return zstd.ZstdDecompressor().stream_reader(fh)


def _main() -> None:
    from stockfish import Stockfish

    parser = argparse.ArgumentParser(
        description="Добирает не-матовые задачи в puzzles.json, чтобы снизить долю матов"
    )
    parser.add_argument("--source", default=os.environ.get("PUZZLES_SOURCE", DEFAULT_SOURCE))
    parser.add_argument("--sf-path", default=os.environ.get("SF_PATH", DEFAULT_SF))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "puzzles.json"))
    parser.add_argument("--bands", default=DEFAULT_BANDS)
    parser.add_argument("--candidates-per-band", type=int, default=500)
    parser.add_argument("--min-scored", type=int, default=100)
    parser.add_argument("--max-per-theme", type=int, default=8)
    parser.add_argument("--max-add-per-band", type=int, default=60)
    parser.add_argument("--depth", type=int, default=14)
    args = parser.parse_args()

    bands = _parse_bands(args.bands)

    if not os.path.exists(args.source):
        raise SystemExit(f"Не найден дамп Lichess: {args.source}")

    with open(args.out, encoding="utf-8") as fh:
        payload = json.load(fh)
    existing = payload.get("puzzles", [])
    existing_ids = {str(p.get("id", "")) for p in existing}
    print(f"[Balanced] В базе уже {len(existing)} задач", flush=True)

    plan: dict[tuple[int, int], int] = {}
    report: dict[tuple[int, int], tuple[int, int]] = {}
    for lo, hi, target, min_add in bands:
        pool = [p for p in existing if lo <= p.get("rating", 0) < hi]
        mates = sum(1 for p in pool if p.get("solver_mates") or _is_mate(p.get("themes", [])))
        report[(lo, hi)] = (len(pool), mates)
        needed = 0
        if target > 0 and mates:
            needed = max(0, math.ceil(mates / target - len(pool)))
        want = max(needed, min_add)
        want = min(want, args.max_add_per_band)
        plan[(lo, hi)] = want
        print(
            f"[Balanced] {lo}-{hi}: задач={len(pool)} матов={mates} "
            f"({100 * mates / len(pool) if pool else 0:.0f}%) -> нужно добавить {want}",
            flush=True,
        )

    wanted_bands = {k for k, v in plan.items() if v > 0}
    if not wanted_bands:
        print("[Balanced] Добавлять нечего — доли матов уже в норме", flush=True)
        return

    sf = Stockfish(
        path=args.sf_path,
        depth=args.depth,
        parameters={"Threads": 1, "Hash": 128},
    )

    buckets: dict[tuple[int, int], list[dict]] = {b: [] for b in wanted_bands}
    t0 = time.time()
    with open(args.source, "rb") as fh, _zstd_reader(fh) as reader:
        reader = io.TextIOWrapper(reader, encoding="utf-8")
        for row in csv.DictReader(reader):
            fen = (row.get("FEN") or "").strip()
            moves = (row.get("Moves") or "").strip()
            if not fen or not moves:
                continue
            try:
                rating = int(float((row.get("Rating") or "").strip()))
            except ValueError:
                continue
            themes = (row.get("Themes") or "").split()
            if _is_mate(themes):
                continue
            band = None
            for (lo, hi) in buckets:
                if lo <= rating < hi:
                    band = (lo, hi)
                    break
            if band is None or len(buckets[band]) >= args.candidates_per_band:
                continue
            pid = (row.get("PuzzleId") or "").strip()
            if not pid or pid in existing_ids:
                continue
            candidate = {"id": pid, "fen": fen, "moves": moves, "rating": rating, "themes": themes}
            ok, _ = _cheap_valid(candidate)
            if ok:
                buckets[band].append(candidate)
            if all(len(v) >= args.candidates_per_band for v in buckets.values()):
                break

    print(
        f"[Balanced] Предкандидатов: {sum(len(v) for v in buckets.values())} "
        f"за {time.time() - t0:.0f}s",
        flush=True,
    )

    added: list[dict] = []
    for band, want in plan.items():
        if want <= 0:
            continue
        lo, hi = band
        pool = list(buckets.get(band, []))
        random.shuffle(pool)
        scored: list[tuple[dict, dict, str]] = []
        stop_after = max(want * 2, args.min_scored)
        for i, p in enumerate(pool, 1):
            if len(scored) >= stop_after:
                break
            if i % 25 == 0:
                print(f"  [{lo}-{hi}] проверено {i}/{len(pool)}...", flush=True)
            try:
                metrics = _validate(p, sf)
            except Exception:
                continue
            if metrics is None or metrics["solver_mates"]:
                continue
            scored.append((p, metrics, _group_of(p, metrics)))

        scored.sort(key=lambda item: -item[1]["first_adv"])

        groups: dict[str, list] = {}
        for item in scored:
            groups.setdefault(item[2], []).append(item)
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))

        picked: list = []
        while len(picked) < want:
            progressed = False
            for theme, group in ordered:
                if len(picked) >= want:
                    break
                if not group:
                    continue
                if sum(1 for _, _, tt in picked if tt == theme) >= args.max_per_theme:
                    continue
                picked.append(group.pop(0))
                progressed = True
            if not progressed:
                break

        for p, m, theme in picked:
            d = _decode(p["fen"], p["moves"])
            added.append(
                {
                    "id": p["id"],
                    "fen": d["presented"],
                    "moves": " ".join(d["player_moves"]),
                    "rating": p["rating"],
                    "themes": p["themes"],
                    "win_score": m["end_solver"],
                    "first_adv": m["first_adv"],
                    "solver_mates": m["solver_mates"],
                    "primary_theme": theme,
                }
            )
        print(f"  корзина {lo}-{hi}: валидных={len(scored)} добавлено={len(picked)}", flush=True)

    merged = existing + added
    merged.sort(key=lambda p: p["rating"])
    out = {
        "source": payload.get("source", "Lichess Puzzles"),
        "count": len(merged),
        "puzzles": merged,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))

    print(f"[Balanced] Добавлено задач: {len(added)}")
    for lo, hi, target, _min in bands:
        n, mates = report[(lo, hi)]
        now = [p for p in merged if lo <= p.get("rating", 0) < hi]
        now_mates = sum(1 for p in now if p.get("solver_mates") or _is_mate(p.get("themes", [])))
        print(
            f"[Balanced] {lo}-{hi}: было {mates}/{n} ({100 * mates / n if n else 0:.0f}%) -> "
            f"стало {now_mates}/{len(now)} "
            f"({100 * now_mates / len(now) if now else 0:.0f}%), цель {target * 100:.0f}%",
            flush=True,
        )
    print(f"[Balanced] Итого в базе: {len(merged)}")
    print(f"[Balanced] Время: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    _main()
