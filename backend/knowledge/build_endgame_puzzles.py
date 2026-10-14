"""Дополняет backend/knowledge/puzzles.json эндшпильными задачами из дампа Lichess.

В отличие от build_puzzles.py этот скрипт НЕ пересобирает базу целиком, а только
добирает новые задачи с эндшпильными темами и подмешивает их к существующим,
не затрагивая уже отобранные пазлы (дедупликация по PuzzleId).

Запуск:
    python -m backend.knowledge.build_endgame_puzzles --add-per-bucket 50

Источник — официальная база database.lichess.org (тот же формат, что и в
build_puzzles.py: FEN до хода соперника, Moves содержит ход соперника первым).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import time

import chess

from backend.knowledge.build_puzzles import (
    BUCKETS,
    _cheap_valid,
    _decode,
    _primary_theme,
    _validate,
)

DEFAULT_SOURCE = r"C:\Users\ASUS\AppData\Local\Temp\opencode\lichess_db_puzzle.csv.zst"
DEFAULT_SF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "stockfish_engine", "stockfish.exe")
)

# Темы, по которым Lichess относит задачу к эндшпилю.
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


def _is_endgame(themes: list[str]) -> bool:
    return bool(set(themes) & ENDGAME_THEMES)


def _main() -> None:
    from stockfish import Stockfish

    parser = argparse.ArgumentParser(description="Добирает эндшпильные пазлы в puzzles.json")
    parser.add_argument("--source", default=os.environ.get("PUZZLES_SOURCE", DEFAULT_SOURCE))
    parser.add_argument("--sf-path", default=os.environ.get("SF_PATH", DEFAULT_SF))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "puzzles.json"))
    parser.add_argument("--candidates-per-bucket", type=int, default=200)
    parser.add_argument("--add-per-bucket", type=int, default=50)
    parser.add_argument("--max-per-theme", type=int, default=20)
    parser.add_argument("--depth", type=int, default=14)
    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise SystemExit(f"Не найден дамп Lichess: {args.source}")

    with open(args.out, encoding="utf-8") as fh:
        payload = json.load(fh)
    existing = payload.get("puzzles", [])
    existing_ids = {str(p.get("id", "")) for p in existing}
    print(f"[Endgames] В базе уже {len(existing)} задач", flush=True)

    sf = Stockfish(
        path=args.sf_path,
        depth=args.depth,
        parameters={"Threads": 1, "Hash": 128},
    )

    buckets = {b: [] for b in BUCKETS}
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
            if not _is_endgame(themes):
                continue
            pid = (row.get("PuzzleId") or "").strip()
            if not pid or pid in existing_ids:
                continue
            bucket = None
            for (lo, hi) in BUCKETS:
                if lo <= rating < hi:
                    bucket = (lo, hi)
                    break
            if bucket is None or len(buckets[bucket]) >= args.candidates_per_bucket:
                continue
            candidate = {"id": pid, "fen": fen, "moves": moves, "rating": rating, "themes": themes}
            ok, _ = _cheap_valid(candidate)
            if ok:
                buckets[bucket].append(candidate)
            if sum(len(v) for v in buckets.values()) % 250 == 0:
                print(f"  ... предкандидаты: {sum(len(v) for v in buckets.values())}", flush=True)
            if all(len(v) >= args.candidates_per_bucket for v in buckets.values()):
                break

    print(
        f"[Endgames] Предкандидатов: {sum(len(v) for v in buckets.values())} за {time.time()-t0:.0f}s",
        flush=True,
    )

    added: list[dict] = []
    for (lo, hi), pool in buckets.items():
        random.shuffle(pool)
        scored: list[tuple[dict, dict, str]] = []
        for i, p in enumerate(pool, 1):
            if i % 25 == 0:
                print(f"  [{lo}-{hi}] проверено {i}/{len(pool)}...", flush=True)
            try:
                metrics = _validate(p, sf)
            except Exception:
                continue
            if metrics is None:
                continue
            theme = "mate" if metrics["solver_mates"] else _primary_theme(p)
            scored.append((p, metrics, theme))

        scored.sort(key=lambda item: -item[1]["first_adv"])

        groups: dict[str, list] = {}
        for item in scored:
            groups.setdefault(item[2], []).append(item)

        picked: list = []
        while len(picked) < args.add_per_bucket:
            progressed = False
            for theme in list(groups.keys()):
                if len(picked) >= args.add_per_bucket:
                    break
                g = groups[theme]
                if not g:
                    continue
                if sum(1 for _, _, tt in picked if tt == theme) >= args.max_per_theme:
                    continue
                picked.append(g.pop(0))
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

    endgames_total = sum(1 for p in merged if _is_endgame(p.get("themes", [])))
    print(f"[Endgames] Добавлено задач: {len(added)}")
    print(f"[Endgames] Итого в базе: {len(merged)} | с эндшпильной темой: {endgames_total}")
    print(f"[Endgames] Время: {time.time()-t0:.0f}s")


def _zstd_reader(fh):
    import zstandard as zstd

    return zstd.ZstdDecompressor().stream_reader(fh)


if __name__ == "__main__":
    _main()
