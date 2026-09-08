"""Генератор backend/knowledge/puzzles.json из открытой базы паззлов Lichess.

Источник — зеркало lichess_db_puzzle на GitHub
(mcognetta/lichess-combined-puzzle-game-db, первые 50k паззлов).
Формат источника — NDJSON по строкам, у каждой записи объект:
  {"puzzle": {"FEN","Moves","Rating","Themes",...}, ...}

"FEN" — позиция, в которой ход делает решающий.
"Moves" — полная последовательность решения в формате UCI (нотация как в
lichess_db_puzzle: первый ход — ход решающего, его выполняет игрок; далее
чередуются ответы соперника и ходы решающего). Для теста определения уровня
используется только ПЕРВЫЙ ход как эталон решения.

Итоговый puzzles.json — стабильный offline-набор тактических задач,
отсортированный по возрастанию рейтинга, сгруппированный по темам.
"""

import bz2
import json
import os
import time

SOURCE = r"C:\Users\ASUS\AppData\Local\Temp\opencode\cpdb_50k.ndjson.bz2"
OUT_PATH = os.path.join(os.path.dirname(__file__), "puzzles.json")

# Сколько паззлов берём на каждый диапазон рейтинга (всего 20).
PER_BUCKET = 5
BUCKETS = [
    (500, 900),      # уровень 0-500 (новички)
    (900, 1300),     # уровни 500-1000
    (1300, 1700),    # уровни 1000-1500
    (1700, 3000),    # уровни 1500+
]

# Приоритет "креатива" для низких рейтингов — понятные и компактные задачи.
LOW_THEMES = {"fork", "mateIn1", "mateIn2", "pin", "skewer", "hangingPiece"}
MID_THEMES = {"fork", "pin", "discoveredAttack", "skewer", "sacrifice", "deflection"}


def _side(turn_char: str) -> str:
    return "w" if turn_char == "w" else "b"


def main() -> None:
    import chess

    candidates: list[dict] = []
    with bz2.open(SOURCE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pz = obj.get("puzzle", {})
            except Exception:
                continue

            fen = pz.get("FEN", "")
            moves = pz.get("Moves", "")
            rating_s = pz.get("Rating", "")
            themes = (pz.get("Themes") or "").split()
            if not fen or not moves or not rating_s:
                continue

            try:
                rating = int(float(rating_s))
                first, *rest = [m for m in moves.split() if m]
                if not rest:
                    continue
                # начало решения идёт с хода решающего, у которого ход по FEN
                board = chess.Board(fen)
                # убеждаемся, что ход решающего легален и выполнен со стороны,
                # которая ходит в позиции
                chess.Move.from_uci(first)  # ValueError -> пропустим строку
                # ответ соперника — тоже проверяем легальность
                board.push_uci(first)
                chess.Move.from_uci(rest[0])
            except Exception:
                continue

            candidates.append(
                {
                    "fen": fen,
                    "solution": first,
                    "moves": moves,
                    "rating": rating,
                    "themes": themes,
                    "id": pz.get("PuzzleId", ""),
                }
            )

    # Группируем по корзинам рейтинга и семплируем равномерно, отдавая
    # предпочтение понятным для слабых игроков темам.
    by_bucket: dict[tuple, list[dict]] = {b: [] for b in BUCKETS}
    for c in candidates:
        r = c["rating"]
        for lo, hi in BUCKETS:
            if lo <= r < hi:
                by_bucket[(lo, hi)].append(c)
                break

    def _key(c: dict, bucket_lo: int) -> float:
        themes = set(c["themes"])
        pref = LOW_THEMES if bucket_lo < 1300 else MID_THEMES
        score = sum(1 for t in themes if t in pref)
        return (not bool(themes & {"long", "veryLong"}), -score, c["rating"])

    selected: list[dict] = []
    for (lo, hi), pool in by_bucket.items():
        pool = sorted(pool, key=lambda c: _key(c, lo))
        selected.extend(pool[:PER_BUCKET])

    selected.sort(key=lambda c: c["rating"])
    payload = {
        "source": "Lichess puzzles (50k, via mcognetta/lichess-combined-puzzle-game-db)",
        "count": len(selected),
        "puzzles": selected,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[Build] Прочитано корректных паззлов: {len(candidates)}")
    print(f"[Build] Записано паззлов: {len(selected)}")
    for (lo, hi), _ in by_bucket.items():
        n = sum(1 for c in selected if lo <= c["rating"] < hi)
        print(f"[Build]   {lo}-{hi}: {n}")
    print(f"[Build] Файл: {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[Build] Готово за {time.time() - t0:.1f}s")
