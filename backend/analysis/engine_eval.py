"""Оценка позиции через Stockfish (общая для разбора и записи партий).

``position_eval`` возвращает оценку в сантипешках от лица белых и мат
от лица белых. Мат сводится к ``±MATE_CP``: для шкалы и точности важно
только то, что позиция решена, а не число ходов до мата.
"""

from __future__ import annotations

import chess

from backend.analysis.win_probability import MATE_CP
from backend.api_gateway.state import stockfish_lock

# Сколько времени движок думает над одной позицией, мс. Один полуход -
# один поиск, поэтому разбор 80 полуходов занимает около 24 секунд.
DEFAULT_SEARCH_MS = 300


def ensure_single_line(engine) -> None:
    """Ставит движку MultiPV = 1 перед серией оценок.

    ``get_evaluation`` возвращает оценку из последней строки ``info``.
    Если у движка включён MultiPV, последняя строка - это худший из
    показанных ходов, и оценка позиции уезжает на несколько пешек.
    """
    multipv: int | None = None

    try:
        multipv = int(engine.get_engine_parameters().get("MultiPV", 1))
    except Exception:  # pragma: no cover - зависит от версии обёртки
        try:
            multipv = int(engine.get_parameters().get("MultiPV", 1))
        except Exception:
            multipv = None

    if multipv == 1:
        return

    try:
        engine.update_engine_parameters({"MultiPV": 1})
    except Exception:  # pragma: no cover - зависит от версии обёртки
        try:
            engine._set_option("MultiPV", 1)
            engine._is_ready()
        except Exception:
            pass


def position_eval(
    engine,
    board: chess.Board,
    search_time_ms: int = DEFAULT_SEARCH_MS,
) -> tuple[float, int | None]:
    """Оценка позиции: (сантипешки от лица белых, мат от лица белых).

    Мат приходит числом ходов: положительное - мат ставят белые,
    отрицательное - чёрные, 0 - на доске уже мат. ``None`` - мата нет.
    Для расчётов мат равен ``±MATE_CP``.

    Завершённые позиции не отправляем в движок: на мате движок всё равно
    не даёт содержательной оценки, а на ничьей она равна нулю.
    """
    if board.is_checkmate():
        return (MATE_CP, 0) if board.turn == chess.BLACK else (-MATE_CP, 0)
    if board.is_game_over(claim_draw=False):
        return 0.0, None

    with stockfish_lock:
        engine.set_fen_position(board.fen())
        evaluation = engine.get_evaluation(searchtime=search_time_ms)

    sign = 1.0 if board.turn == chess.WHITE else -1.0

    if evaluation and evaluation.get("type") == "mate":
        mate = int(evaluation.get("value") or 0)
        direction = 1.0 if mate > 0 else -1.0
        return direction * sign * MATE_CP, mate * int(sign)

    cp = float(evaluation.get("value") or 0) * sign
    return cp, None
