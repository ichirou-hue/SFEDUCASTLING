"""
NAG Labeler (025) — быстрая векторизованная версия.
Добавляет NAG-метки (!, ?, !!, ??, !?, ?!) к каждому ходу.
"""

import logging
import chess
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def cpl_from_evals(eval_before: pd.Series, eval_after: pd.Series) -> pd.Series:
    """Centipawn loss для игрока, который только что сходил."""
    # cpl = max(0, -eval_after - eval_before)
    # eval_after нужно флипнуть, так как очередь сменилась
    cpl = (-eval_after - eval_before).clip(lower=0)
    return cpl


def classify_nag(cpl: pd.Series, is_best: pd.Series) -> pd.Series:
    """Векторизованная классификация NAG по стандартам chess.com / Lichess."""
    nag = pd.Series("", index=cpl.index, dtype=object)

    # блундер
    nag = nag.mask(cpl > 300, "??")
    # ошибка
    nag = nag.mask((cpl > 100) & (cpl <= 300), "?")
    # неточность
    nag = nag.mask((cpl > 20) & (cpl <= 100), "?!")
    # хороший ход
    nag = nag.mask((cpl > 0) & (cpl <= 20), "!?")
    # лучший ход (cpl == 0)
    nag = nag.mask((cpl == 0) & is_best, "!!")
    # cpl == 0 но не лучший (другой равноценный ход)
    nag = nag.mask((cpl == 0) & (~is_best), "!")

    return nag


def san_to_uci_vectorized(fen_series: pd.Series, san_series: pd.Series) -> pd.Series:
    """Векторизованная конвертация SAN -> UCI."""
    def _to_uci(fen, san):
        try:
            board = chess.Board(fen)
            return board.parse_san(san).uci()
        except Exception:
            return None

    return pd.Series([_to_uci(f, s) for f, s in zip(fen_series, san_series)], index=fen_series.index)


def add_nag_labels(
    input_path: str | Path,
    output_path: str | Path,
    limit: Optional[int] = None,
) -> None:
    """
    Добавляет колонку `nag` к датасету с eval_cp и best_move_uci.
    Векторизованная версия — без циклов, работает за секунды на 1М позиций.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit)
    logger.info("Добавляю NAG к %d позициям (векторизованно)...", len(df))

    required = ["fen", "san", "eval_cp", "best_move_uci", "game_id", "ply"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Нет колонок: {missing}")

    # 1. SAN -> UCI векторизованно
    df["move_uci"] = san_to_uci_vectorized(df["fen"], df["san"])

    # 2. Флаг "сходил лучший ход"
    df["is_best"] = df["move_uci"] == df["best_move_uci"]

    # 3. eval_after через shift внутри каждой партии
    df = df.sort_values(["game_id", "ply"])
    df["eval_after"] = df.groupby("game_id")["eval_cp"].shift(-1)

    # 4. Centipawn loss
    df["cpl"] = cpl_from_evals(df["eval_cp"], df["eval_after"])

    # 4. Классификация NAG
    df["nag"] = classify_nag(df["cpl"], df["is_best"])

    # Чистка временных колонок
    drop_cols = ["move_uci", "is_best", "eval_after", "cpl"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    vc = df["nag"].value_counts()
    logger.info("025: готово %d позиций за %s → %s", len(df), "векторизованно", output_path)
    logger.info("Распределение NAG:\n%s", vc.to_string())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="NAG Labeler (025) — векторизованный")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Ограничить для теста")
    args = parser.parse_args()
    add_nag_labels(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()