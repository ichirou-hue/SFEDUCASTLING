"""
Stockfish Labeler — добавляет eval_cp и best_move в датасет.

Запуск:
  python -m training_data.labeler.stockfish_labeler
"""

import os
import logging
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import pandas as pd

logger = logging.getLogger(__name__)

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
DEPTH = int(os.getenv("STOCKFISH_DEPTH", "18"))
THREADS = int(os.getenv("STOCKFISH_THREADS", "2"))
HASH_MB = int(os.getenv("STOCKFISH_HASH", "256"))


def get_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": THREADS, "Hash": HASH_MB})
    return engine


def analyze_fen(engine: chess.engine.SimpleEngine, fen: str, depth: int) -> dict:
    """Анализирует одну позицию, возвращает eval_cp и best_move_uci."""
    board = chess.Board(fen)
    if board.is_game_over():
        return {"eval_cp": None, "best_move_uci": None, "depth": depth}

    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info.get("score")
    pv = info.get("pv", [])

    eval_cp = None
    if score is not None:
        eval_cp = score.relative.score(mate_score=10000)

    best_move_uci = pv[0].uci() if pv else None

    return {"eval_cp": eval_cp, "best_move_uci": best_move_uci, "depth": depth}


def label_dataset(
    input_path: str | Path,
    output_path: str | Path,
    limit: Optional[int] = None,
    depth: int = DEPTH,
) -> None:
    """
    Читает Parquet с FEN, прогоняет через Stockfish, сохраняет с eval_cp + best_move_uci.

    Args:
        input_path: Путь к исходному .parquet (колонка 'fen' обязательна).
        output_path: Куда сохранить размеченный .parquet.
        limit: Ограничить количество позиций (для теста).
        depth: Глубина анализа Stockfish.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit)

    logger.info("Размечаю %d позиций (depth=%d)...", len(df), depth)

    engine = get_engine()
    evals = []
    best_moves = []

    try:
        for i, fen in enumerate(df["fen"]):
            res = analyze_fen(engine, fen, depth)
            evals.append(res["eval_cp"])
            best_moves.append(res["best_move_uci"])

            if (i + 1) % 100 == 0:
                logger.info("  %d/%d", i + 1, len(df))
    finally:
        engine.quit()

    df["eval_cp"] = evals
    df["best_move_uci"] = best_moves
    df.to_parquet(output_path, index=False)
    logger.info("Сохранено в %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Stockfish Labeler (024)")
    parser.add_argument("--input", default="training_data/labeled/dataset_final.parquet")
    parser.add_argument("--output", default="training_data/labeled/dataset_labeled.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить для теста")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Глубина анализа")
    args = parser.parse_args()

    label_dataset(args.input, args.output, args.limit, args.depth)


if __name__ == "__main__":
    main()