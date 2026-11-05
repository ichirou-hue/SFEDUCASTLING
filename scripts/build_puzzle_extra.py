"""Сборка backend/knowledge/puzzle_extra.json из выгрузки Lichess.

Выгрузка (``extra_candidates.json``) повторяет формат официальной базы:
FEN - позиция ПЕРЕД ходом соперника, Moves - полная линия вместе с этим
ходом. Для курса нужен уже раскодированный вид, как в ``puzzles.json``:

    fen   = позиция после хода соперника (ход у решающего);
    moves = оставшаяся линия, начиная с хода решающего.

Запуск::

    python scripts/build_puzzle_extra.py --source путь/к/extra_candidates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess

from backend.knowledge.build_puzzles import _decode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "backend" / "knowledge" / "puzzle_extra.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    raw = json.loads(Path(args.source).read_text(encoding="utf-8"))
    rows = [row for items in raw.values() for row in items]

    seen: set[str] = set()
    puzzles: list[dict] = []
    for row in rows:
        key = row.get("id") or row.get("fen")
        if key in seen:
            continue
        seen.add(key)
        decoded = _decode(row["fen"], row["moves"])
        if decoded is None:
            continue
        # Проверяем, что линия действительно доигрывается до конца.
        board = chess.Board(decoded["presented"])
        try:
            for uci in decoded["player_moves"]:
                board.push_uci(uci)
        except ValueError:
            continue
        puzzles.append(
            {
                "id": row.get("id"),
                "fen": decoded["presented"],
                "moves": " ".join(decoded["player_moves"]),
                "rating": int(row.get("rating") or 0),
                "themes": list(row.get("themes") or []),
            }
        )

    puzzles.sort(key=lambda item: (item["rating"], item["id"] or ""))
    payload = {
        "source": "Lichess Puzzles (official database.lichess.org), доп. темы",
        "count": len(puzzles),
        "puzzles": puzzles,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Записано {len(puzzles)} задач в {args.out}")


if __name__ == "__main__":
    main()
