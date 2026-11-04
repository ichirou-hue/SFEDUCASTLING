"""Фоновая запись завершённых партий (задача 83).

Принимает полную партию (список ходов UCI + результат), проигрывает её,
оценивает позицию после каждого полухода через Stockfish и классифицирует
ход по падению вероятности победы - той же моделью, что использует разбор
партии. Оценки и классы сохраняются в games + game_moves.

Раньше ход сравнивался с отдельным поиском «лучшего хода», а класс
ставился по сантипешкам (20/50/100/300/500 cp). Из-за этого в уже
проигранной позиции любой ход выглядел «лучшим», и точность по этапам
в профиле была завышена. Теперь достаточно одной оценки на полуход:
соседние оценки и дают потерю игрока.

Запускается через FastAPI BackgroundTasks после ответа клиенту,
чтобы длительная оценка Stockfish (0.5 с на ход) не блокировала запрос.
"""

import chess
from fastapi.concurrency import run_in_threadpool

from backend.analysis.classifier import classify_move
from backend.analysis.engine_eval import ensure_single_line, position_eval
from backend.api_gateway.state import ensure_stockfish, stockfish_lock
from backend.db.session import async_session_factory
from backend.models.game import Game, GameMove

# Время поиска на полуход в фоновой записи. Разбор партии использует
# 300 мс, здесь чуть больше: запись идёт в фоне и не мешает клиенту.
SEARCH_MS = 500


def _classify_game(moves: list[str]) -> list[dict]:
    """Проигрывает партию и оценивает каждый ход.

    Оценка берётся один раз на позицию: стартовая позиция, затем позиция
    после каждого полухода. Потеря хода - разница соседних оценок от лица
    стороны, делавшей ход; класс - по падению вероятности победы.

    Returns:
        Список записей ходов: fen, uci, san, turn, classification,
        ev_before, ev_after (сантипешки от лица белых), mate.
    """
    board = chess.Board()
    try:
        engine = ensure_stockfish()
    except Exception as exc:
        print(f"[Game analysis] Stockfish unavailable; saving moves without scores: {exc}")
        engine = None
    records: list[dict] = []

    prev_cp: float | None = None
    if engine is not None:
        try:
            with stockfish_lock:
                ensure_single_line(engine)
            prev_cp, _ = position_eval(engine, board, SEARCH_MS)
        except Exception as exc:
            print(f"[Game analysis] Stockfish failed; saving moves without scores: {exc}")
            engine = None

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
        board.push(move)

        cp: float | None = None
        mate: int | None = None
        if engine is not None:
            try:
                cp, mate = position_eval(engine, board, SEARCH_MS)
            except Exception as exc:
                print(f"[Game analysis] Stockfish failed; saving remaining moves without scores: {exc}")
                engine = None

        classification = None
        diff_cp = None
        if cp is not None and prev_cp is not None:
            classification = classify_move(prev_cp, cp, turn)
            # Потеря в сантипешках от лица ходившей стороны: пригодится
            # для отладки и старых клиентов.
            sign = 1.0 if turn == chess.WHITE else -1.0
            diff_cp = round(max(0.0, (prev_cp - cp) * sign), 1)

        records.append({
            "fen": fen_before,
            "uci": uci,
            "san": san,
            "turn": "w" if turn == chess.WHITE else "b",
            "classification": classification,
            "ev_before": prev_cp,
            "ev_after": cp,
            "mate": mate,
            "diff_cp": diff_cp,
        })

        if cp is not None:
            prev_cp = cp

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
                    ev_before=rec["ev_before"],
                    ev_after=rec["ev_after"],
                    diff_cp=rec["diff_cp"],
                    eval_raw={
                        "model": "win_probability",
                        "mate": rec["mate"],
                    },
                )
            )

        await db.commit()
