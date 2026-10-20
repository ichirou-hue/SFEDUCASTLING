# ruff: noqa: T201
"""Пополнение backend/knowledge/puzzles.json из официальной базы Lichess.

База лежит одним файлом (lichess_db_puzzle.csv.zst, ~1 ГБ). Качать её
целиком не нужно: потоки читаются по порядку, а записи в базе
перемешаны, поэтому достаточно первых десятков мегабайт, чтобы набрать
случайную выборку по всем рейтинговым корзинам. Как только в каждой
корзине набрано достаточно кандидатов (или достигнут лимит загрузки),
чтение обрывается.

Отбор:
- ходы легальны, игрок не матан/не под шахом (как в build_puzzles);
- Stockfish подтверждает перевес игрока (win_score/first_adv);
- тема, называющая фигуру, действительно содержит ход этой фигуры
  (puzzle_validation.validate_puzzle).

Существующие пазлы сохраняются, новые добавляются по id без дублей.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time

import requests
import zstandard as zstd

from backend.knowledge.build_puzzles import (
    _DEFAULT_SF,
    BUCKETS,
    DEPTH,
    LESSON_THEMES,
    _cheap_valid,
    _decode,
    _primary_theme,
    _validate,
)
from backend.services.puzzle_validation import validate_puzzle

OFFICIAL_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
_DEFAULT_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "puzzles.json"))


class _ByteCounter:
    """Обёртка потока с подсчётом прочитанных байт."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        data = self._fh.read(size)
        self.count += len(data)
        return data


def _iter_rows(source: str, max_bytes: int):
    """Отдаёт (row, downloaded_bytes), прерывая поток по лимиту байт."""
    is_url = source.startswith("http")
    response = None
    handle = None
    if is_url:
        response = requests.get(source, stream=True, timeout=120)
        response.raise_for_status()
        counter = _ByteCounter(response.raw)
    else:
        handle = open(source, "rb")  # noqa: SIM115 - закрывается в finally
        counter = _ByteCounter(handle)

    try:
        with zstd.ZstdDecompressor().stream_reader(counter) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for row in csv.DictReader(text):
                yield row, counter.count
                if max_bytes and counter.count > max_bytes:
                    break
    finally:
        if response is not None:
            response.close()
        if handle is not None:
            handle.close()


def _to_candidate(row: dict) -> dict | None:
    fen = (row.get("FEN") or "").strip()
    moves = (row.get("Moves") or "").strip()
    try:
        rating = int(float((row.get("Rating") or "").strip()))
    except ValueError:
        return None
    if not fen or not moves:
        return None
    return {
        "id": (row.get("PuzzleId") or "").strip(),
        "fen": fen,
        "moves": moves,
        "rating": rating,
        "themes": (row.get("Themes") or "").split(),
    }


def _bucket_for(rating: int) -> tuple[int, int] | None:
    for lo, hi in BUCKETS:
        if lo <= rating < hi:
            return (lo, hi)
    return None


def _collect(
    source: str,
    per_bucket: int,
    max_bytes: int,
    skip_rows: int,
) -> dict[tuple[int, int], list[dict]]:
    target = per_bucket * 5
    buckets: dict[tuple[int, int], list[dict]] = {b: [] for b in BUCKETS}
    seen = 0
    last_log = 0.0
    for row, downloaded in _iter_rows(source, max_bytes):
        seen += 1
        if seen <= skip_rows:
            if seen % 20000 == 0:
                print(f"  пропущено {seen} строк, скачано {downloaded // 1024} КБ", flush=True)
            continue
        candidate = _to_candidate(row)
        if candidate is None:
            continue
        bucket = _bucket_for(candidate["rating"])
        if bucket is None or len(buckets[bucket]) >= target:
            if all(len(buckets[b]) >= target for b in BUCKETS):
                break
            continue
        ok, _ = _cheap_valid(candidate)
        if not ok:
            continue
        if not validate_puzzle(_project_form(candidate)):
            continue
        buckets[bucket].append(candidate)
        now = time.time()
        if now - last_log > 5:
            got = sum(len(v) for v in buckets.values())
            print(f"  кандидатов {got}, скачано {downloaded // 1024} КБ", flush=True)
            last_log = now
    return buckets


def _project_form(puzzle: dict) -> dict:
    """Переводит запись Lichess (FEN до хода соперника) в формат проекта."""
    decoded = _decode(puzzle["fen"], puzzle["moves"])
    if not decoded:
        return {"fen": None, "moves": None, "themes": puzzle.get("themes")}
    return {
        "fen": decoded["presented"],
        "moves": " ".join(decoded["player_moves"]),
        "themes": puzzle.get("themes"),
    }


def _pick(
    pool: list[dict],
    sf,
    per_bucket: int,
    max_per_theme: int,
) -> list[dict]:
    picked: list = []
    theme_counts: dict[str, int] = {}
    for candidate in pool:
        if len(picked) >= per_bucket:
            break
        try:
            metrics = _validate(candidate, sf)
        except Exception:
            continue
        if metrics is None:
            continue
        theme = "mate" if metrics["solver_mates"] else _primary_theme(candidate)
        if theme in LESSON_THEMES and theme_counts.get(theme, 0) >= max_per_theme:
            continue
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
        picked.append((candidate, metrics, theme))

    result = []
    for candidate, metrics, theme in picked:
        decoded = _decode(candidate["fen"], candidate["moves"])
        result.append(
            {
                "id": candidate["id"],
                "fen": decoded["presented"],
                "moves": " ".join(decoded["player_moves"]),
                "rating": candidate["rating"],
                "themes": candidate["themes"],
                "win_score": metrics["end_solver"],
                "first_adv": metrics["first_adv"],
                "solver_mates": metrics["solver_mates"],
                "primary_theme": theme,
            }
        )
    return result


def main() -> None:
    from stockfish import Stockfish

    parser = argparse.ArgumentParser(description="Пополнить puzzles.json из базы Lichess")
    parser.add_argument("--source", default=OFFICIAL_URL)
    parser.add_argument("--sf-path", default=os.environ.get("SF_PATH", _DEFAULT_SF))
    parser.add_argument("--out", default=_DEFAULT_OUT)
    parser.add_argument("--per-bucket", type=int, default=12)
    parser.add_argument("--max-per-theme", type=int, default=3)
    parser.add_argument("--max-mb", type=int, default=80)
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=100_000,
        help="Пропустить первые строки базы (существующие пазлы берутся из начала файла)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Всего добавить (0 — без лимита)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[Fetch] Источник: {args.source}", flush=True)
    buckets = _collect(
        args.source, args.per_bucket, args.max_mb * 1024 * 1024, args.skip_rows
    )
    for bucket, pool in buckets.items():
        print(f"  корзина {bucket[0]}-{bucket[1]}: кандидатов {len(pool)}", flush=True)

    sf = Stockfish(path=args.sf_path, depth=DEPTH, parameters={"Threads": 1, "Hash": 64})

    added: list[dict] = []
    for bucket in BUCKETS:
        picked = _pick(buckets[bucket], sf, args.per_bucket, args.max_per_theme)
        added.extend(picked)
        print(f"  корзина {bucket[0]}-{bucket[1]}: отобрано {len(picked)}", flush=True)

    if args.limit:
        added = added[: args.limit]

    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = {
            "source": "Lichess Puzzles (official database.lichess.org)",
            "count": 0,
            "puzzles": [],
        }
    existing = payload.get("puzzles", [])
    known = {p["id"] for p in existing}
    fresh = [p for p in added if p["id"] not in known]
    merged = existing + fresh
    merged.sort(key=lambda p: p["rating"])

    payload["count"] = len(merged)
    payload["puzzles"] = merged
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

    print(f"[Fetch] Добавлено {len(fresh)} новых пазлов, всего {len(merged)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
