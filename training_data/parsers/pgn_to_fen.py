"""
PGN → FEN конвертер для обучения ML модели.

Читает JSONL с партиями от Lichess, парсит ходы (PGN/SAN),
генерирует FEN для каждой позиции и сохраняет в Parquet.
"""

import json
import logging
from pathlib import Path
from typing import Iterator

import chess
import pandas as pd

logger = logging.getLogger(__name__)


def iter_positions_from_game(game: dict) -> Iterator[dict]:
    """
    Генерирует позиции (FEN) из одной партии.

    Args:
        game: Словарь с данными игры от Lichess API.
              Ожидаются ключи: id, moves (PGN в SAN), и опционально:
              white, black, result, speed, perf, rated, createdAt, lastMoveAt.

    Yields:
        Словарь для каждой позиции:
        - fen: FEN строка позиции
        - game_id: ID партии
        - ply: номер полухода (0 = начальная позиция)
        - move_number: номер хода (1, 2, 3...)
        - turn: "white" или "black" — чья очередь ходить
        - san: ход в SAN, который привел к этой позиции (пусто для начальной)
    """
    game_id = game.get("id", "")
    moves_str = game.get("moves", "")
    if not moves_str:
        return

    board = chess.Board()

    # Начальная позиция (ply=0)
    yield {
        "fen": board.fen(),
        "game_id": game_id,
        "ply": 0,
        "move_number": 1,
        "turn": "white",
        "san": "",
    }

    ply = 0
    move_number = 1
    for san in moves_str.split():
        try:
            move = board.parse_san(san)
        except ValueError as e:
            logger.warning("Игра %s: ошибка парсинга SAN '%s': %s", game_id, san, e)
            break

        board.push(move)
        ply += 1
        turn = "white" if board.turn == chess.WHITE else "black"

        # Увеличиваем номер хода после черного
        if board.turn == chess.WHITE:
            move_number += 1

        yield {
            "fen": board.fen(),
            "game_id": game_id,
            "ply": ply,
            "move_number": move_number,
            "turn": turn,
            "san": san,
        }


def convert_jsonl(input_path: str | Path, output_path: str | Path) -> Path:
    """
    Конвертирует JSONL с партиями в Parquet с FEN позициями.

    Args:
        input_path: Путь к .jsonl файлу (одна JSON-партия на строку).
        output_path: Путь к выходному .parquet файлу.

    Returns:
        Путь к созданному .parquet файлу.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Конвертирую %s → %s", input_path, output_path)

    rows = []
    game_count = 0
    pos_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            game = json.loads(line)
            game_count += 1

            for pos in iter_positions_from_game(game):
                rows.append(pos)
                pos_count += 1

            if game_count % 100 == 0:
                logger.info("  Обработано %d игр, %d позиций", game_count, pos_count)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    logger.info("Готово: %d игр → %d позиций в %s", game_count, pos_count, output_path)
    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="PGN → FEN конвертер (JSONL → Parquet)")
    parser.add_argument("input", help="Входной .jsonl файл")
    parser.add_argument("output", help="Выходной .parquet файл")
    args = parser.parse_args()
    convert_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()