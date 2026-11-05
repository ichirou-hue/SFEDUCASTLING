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

from backend.analysis.brilliant import brilliant_sacrifice
from backend.analysis.classifier import classify_move
from backend.analysis.engine_eval import ensure_single_line, position_eval
from backend.api_gateway.state import ensure_stockfish, stockfish_lock
from backend.db.session import async_session_factory
from backend.models.game import Game, GameMove

# Время поиска на полуход в фоновой записи. Разбор партии использует
# 300 мс, здесь чуть больше: запись идёт в фоне и не мешает клиенту.
SEARCH_MS = 500
# Extra search only on the human player's turns; keeps background analysis bounded.
BRILLIANT_SEARCH_MS = 250


def _classify_game(moves: list[str], player_color: str = "w") -> list[dict]:
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
    player_turn = chess.WHITE if player_color == "w" else chess.BLACK

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
        is_player_move = turn == player_turn
        board_before = board.copy(stack=False)

        best_uci = None
        if engine is not None and is_player_move:
            try:
                with stockfish_lock:
                    engine.set_fen_position(fen_before)
                    best_uci = engine.get_best_move_time(BRILLIANT_SEARCH_MS)
            except Exception as exc:
                # Best-move metadata is optional: never lose the whole game record.
                print(f"[Game analysis] Brilliant search failed for {uci}: {exc}")

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

        brilliant = {"is_brilliant": False, "sacrifice_value": 0, "reason": None}
        if is_player_move and classification is not None:
            brilliant = brilliant_sacrifice(
                board_before,
                move,
                best_uci=best_uci,
                classification=classification,
            )

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
            "best_uci": best_uci,
            "brilliant": brilliant["is_brilliant"],
            "brilliant_reason": brilliant["reason"],
            "sacrifice_value": brilliant["sacrifice_value"],
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

    white_name = str(white or "").casefold()
    black_name = str(black or "").casefold()
    player_color = "b" if black_name in {"player", "user"} and white_name not in {"player", "user"} else "w"
    classified = await run_in_threadpool(_classify_game, moves, player_color)

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
                        "best_uci": rec["best_uci"],
                        "brilliant": bool(rec["brilliant"]),
                        "brilliant_reason": rec["brilliant_reason"],
                        "sacrifice_value": int(rec["sacrifice_value"] or 0),
                    },
                )
            )

        await db.commit()
