"""Фоновая запись завершённых партий (задача 83).

Принимает полную партию (список ходов UCI + результат), проигрывает её
и для каждого хода считает потерю в центепешках через Stockfish,
классифицирует ход (best/excellent/...) и сохраняет games + game_moves.

Запускается через FastAPI BackgroundTasks после ответа клиенту,
чтобы длительная оценка Stockfish (1с на ход) не блокировала запрос.
"""

import chess
from fastapi.concurrency import run_in_threadpool

from backend.analysis.classifier import classify, normalize_eval
from backend.api_gateway.state import ensure_stockfish, stockfish_lock
from backend.db.session import async_session_factory
from backend.models.game import Game, GameMove


def _classify_game(moves: list[str]) -> list[dict]:
    """Проигрывает партию и классифицирует каждый ход.

    Для каждого хода игрока сравнивается оценка позиции ПОСЛЕ его хода
    с оценкой позиции ПОСЛЕ лучшего хода Stockfish в той же позиции.

    Returns:
        Список записей ходов: fen, uci, san, turn, classification,
        ev_best, ev_played, best_uci, diff_cp.
    """
    board = chess.Board()
    sf = ensure_stockfish()
    records: list[dict] = []

    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
        except ValueError:
            break

        fen_before = board.fen()
        turn = board.turn
        san = board.san(move)

        ev_best = None
        ev_played = None
        best_uci = None

        if sf is not None:
            with stockfish_lock:
                sf.set_fen_position(fen_before)
                best_uci = sf.get_best_move_time(1000)
                if best_uci:
                    try:
                        best_move = chess.Move.from_uci(best_uci)
                        if best_move in board.legal_moves:
                            best_board = board.copy()
                            best_board.push(best_move)
                            sf.set_fen_position(best_board.fen())
                            ev_best = sf.get_evaluation(searchtime=1000)
                    except ValueError:
                        ev_best = None

        board.push(move)

        if sf is not None:
            with stockfish_lock:
                sf.set_fen_position(board.fen())
                ev_played = sf.get_evaluation(searchtime=1000)

        # Потеря = насколько ход игрока хуже лучшего хода (обе оценки
        # приводятся к точке зрения стороны, делавшей ход).
        diff_cp = None
        classification = None
        if ev_best is not None and ev_played is not None:
            # ev_best — оценка позиции, где ход уже перешёл к сопернику,
            # поэтому нормализуем к стороне, которая сделала ход.
            before_side = -normalize_eval(ev_best)
            after_side = -normalize_eval(ev_played)
            diff_cp = before_side - after_side
            classification = classify(diff_cp)

        records.append({
            "fen": fen_before,
            "uci": uci,
            "san": san,
            "turn": "w" if turn == chess.WHITE else "b",
            "classification": classification,
            "ev_best": ev_best,
            "ev_played": ev_played,
            "best_uci": best_uci,
            "diff_cp": diff_cp,
        })

    return records


async def record_game(
    moves: list[str],
    user_id: int | None,
    elo: int | None,
    engine: str,
    result: str,
    status: str,
    white: str = "player",
    black: str = "AI",
) -> None:
    """Создаёт Game + GameMove в БД. Идемпотентна для пустого списка ходов."""
    if not moves:
        return

    classified = await run_in_threadpool(_classify_game, moves)

    async with async_session_factory() as db:
        game = Game(
            user_id=user_id,
            white=white,
            black=black,
            result=result,
            status=status,
            elo=elo,
            engine=engine,
        )
        db.add(game)
        await db.flush()

        for i, rec in enumerate(classified, start=1):
            db.add(
                GameMove(
                    game_id=game.id,
                    move_no=i,
                    fen=rec["fen"],
                    uci=rec["uci"],
                    san=rec["san"],
                    turn=rec["turn"],
                    classification=rec["classification"],
                    ev_before=normalize_eval(rec["ev_best"]),
                    ev_after=normalize_eval(rec["ev_played"]),
                    diff_cp=rec["diff_cp"],
                    eval_raw={"best_uci": rec["best_uci"], "best": rec["ev_best"], "played": rec["ev_played"]},
                )
            )

        await db.commit()
