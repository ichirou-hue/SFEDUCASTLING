"""Игровые endpoint'ы: легальные ходы, выполнение хода, ход AI."""

import random

import chess
from fastapi import APIRouter, BackgroundTasks

from backend.api_gateway.models import CompareMovesRequest, FenRequest, FenSquare, FinishGameRequest, MoveRequest
from backend.api_gateway.state import (
    elo_to_temperature,
    ensure_stockfish,
    get_maia3,
    maia3_lock,
    reset_stockfish,
    stockfish_lock,
)
from backend.services.game_recorder import record_game

router = APIRouter(tags=["game"])


LEVELS = [
    {"level": 1, "name": "Новичок", "elo": 1100, "engine": "maia3"},
    {"level": 2, "name": "Любитель", "elo": 1300, "engine": "maia3"},
    {"level": 3, "name": "Клубный", "elo": 1500, "engine": "maia3"},
    {"level": 4, "name": "Опытный", "elo": 1700, "engine": "maia3"},
    {"level": 5, "name": "Сильный", "elo": 1900, "engine": "maia3"},
    {"level": 6, "name": "Мастер", "elo": 2200, "engine": "maia3"},
    {"level": 7, "name": "Гроссмейстер", "elo": 2600, "engine": "maia3"},
    {"level": 8, "name": "Максимум (Stockfish)", "elo": 3600, "engine": "stockfish"},
]


@router.get("/api/levels")
def get_levels():
    """Возвращает список доступных уровней игры."""
    return {"levels": LEVELS}


@router.post("/api/legal-moves")
def legal_moves(req: FenSquare):
    """Возвращает все легальные ходы для фигуры на указанном поле."""
    board = chess.Board(req.fen)
    sq = chess.parse_square(req.square)
    moves = [
        chess.square_name(move.to_square)
        for move in board.legal_moves
        if move.from_square == sq
    ]
    return {"moves": moves}


@router.post("/api/move")
def make_move(req: MoveRequest):
    """Выполняет шахматный ход и возвращает новую позицию."""
    board = chess.Board(req.fen)
    from_sq = chess.parse_square(req.from_sq)
    to_sq = chess.parse_square(req.to_sq)

    promo = None
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if chess.square_rank(to_sq) in (0, 7):
            promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
            promo = promo_map.get(req.promotion, chess.QUEEN)

    move = chess.Move(from_sq, to_sq, promotion=promo)

    if move not in board.legal_moves:
        return {"error": "Недопустимый ход", "fen": req.fen}

    san = board.san(move)
    board.push(move)

    status = "playing"
    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    elif board.is_check():
        status = "check"

    return {
        "fen": board.fen(),
        "san": san,
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
    }


@router.post("/api/stockfish-analysis")
def stockfish_analysis(req: FenRequest):
    """Анализирует позицию 10 секунд: лучший ход + оценка + топ ходов.

    Фронтенд вызывает этот эндпойнт после каждого хода игрока
    и рисует стрелку от from_sq к to_sq на лучший ход.
    """
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Партия окончена", "fen": req.fen}

    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен", "fen": req.fen}

    try:
        with stockfish_lock:
            print(f"[Analysis] Анализ позиции {req.fen} (10 сек)...")
            stockfish.update_engine_parameters({"UCI_LimitStrength": False})
            stockfish.set_fen_position(req.fen)
            best_uci = stockfish.get_best_move_time(10000)
            evaluation = stockfish.get_evaluation(searchtime=1000)
            print(f"[Analysis] Лучший ход: {best_uci}, оценка: {evaluation}")
    except Exception as e:
        print(f"[Analysis] Ошибка: {e}")
        reset_stockfish()
        return {"error": str(e), "fen": req.fen}

    if not best_uci:
        return {"error": "Не удалось найти ход", "fen": req.fen}

    move = chess.Move.from_uci(best_uci)
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)

    return {
        "fen": req.fen,
        "best_move": best_uci,
        "from": from_name,
        "to": to_name,
        "san": board.san(move),
        "evaluation": evaluation,
    }


@router.post("/api/maia-move")
@router.post("/api/stockfish-move")
def ai_move(req: FenRequest):
    """Получает ход от AI.

    По умолчанию ходит Maia3 — играет как человек заданного Elo
    (req.elo + req.moves для истории партии). Если запрошен engine="stockfish"
    или Maia3 недоступна — ход делает Stockfish на максимальной силе.
    """
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Партия окончена", "fen": req.fen}

    if req.engine == "maia3":
        move = _maia3_move(req, board)
    else:
        move = _stockfish_move(board)

    if move is None:
        print("[AI] Primary engine не вернул ход — fallback на Stockfish")
        move = _stockfish_move(board)

    if move is None:
        move = random.choice(list(board.legal_moves))

    san = board.san(move)
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)
    board.push(move)

    status = "playing"
    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    elif board.is_check():
        status = "check"

    evaluation = None
    if req.engine == "stockfish":
        sf = ensure_stockfish()
        if sf:
            try:
                with stockfish_lock:
                    sf.set_fen_position(board.fen())
                    evaluation = sf.get_evaluation(searchtime=1000)
            except Exception as e:
                print(f"[Stockfish] Ошибка eval: {e}")

    return {
        "fen": board.fen(),
        "san": san,
        "from": from_name,
        "to": to_name,
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
        "evaluation": evaluation,
        "engine": "maia3" if req.engine == "maia3" else "stockfish",
    }


def _maia3_move(req: FenRequest, board: chess.Board) -> chess.Move | None:
    """Ход Maia3: человекоподобная игра на уровне req.elo."""
    engine = get_maia3()
    if engine is None:
        print("[Maia3] Недоступна — fallback на Stockfish")
        return _stockfish_move(board)
    try:
        import chess.engine

        # Восстанавливаем историю партии из ходов (для --use-uci-history).
        history_board = board
        if req.moves:
            try:
                replay = chess.Board()
                for uci in req.moves:
                    replay.push_uci(uci)
                if replay.fen() == board.fen():
                    history_board = replay
                else:
                    print("[Maia3] History FEN mismatch, using board directly")
            except ValueError:
                pass

        with maia3_lock():
            temperature = elo_to_temperature(req.elo)
            print(f"[Maia3] Ход на Elo {req.elo}, temp={temperature}...")
            engine.configure({
                "Elo": req.elo,
                "SelfElo": req.elo,
                "OppoElo": req.elo,
                "Temperature": temperature,
            })
            result = engine.play(history_board, limit=chess.engine.Limit(time=3))
            move = result.move
        print(f"[Maia3] Ход: {move}")
        if move is not None and move in board.legal_moves:
            return move
        if move is not None:
            print(f"[Maia3] Ход {move} не легален в текущей позиции, fallback")
            return _stockfish_move(board)
    except Exception as e:
        print(f"[Maia3] Ошибка хода: {e}")
        try:
            engine.configure({"Elo": 1500, "SelfElo": 1500, "OppoElo": 1500, "Temperature": 1.0})
        except Exception:
            pass
        print("[Maia3] Fallback на Stockfish после ошибки")
        return _stockfish_move(board)
    return None


def _stockfish_move(board: chess.Board) -> chess.Move | None:
    """Ход Stockfish на максимальной силе."""
    stockfish = ensure_stockfish()
    if stockfish:
        try:
            with stockfish_lock:
                print("[AI] Начало расчёта хода...")
                stockfish.update_engine_parameters({"UCI_LimitStrength": False})
                stockfish.set_fen_position(board.fen())
                best_uci = stockfish.get_best_move_time(10000)
                print(f"[AI] Ход: {best_uci}")
            if best_uci:
                return chess.Move.from_uci(best_uci)
        except Exception as e:
            print(f"[Stockfish] Ошибка хода: {e}")
            reset_stockfish()
    else:
        print("[AI] Stockfish недоступен, случайный ход")
    return None


@router.post("/api/compare-moves")
def compare_moves(req: CompareMovesRequest):
    """Сравнивает лучший ход Stockfish с ходом Maia3."""

    board = chess.Board(req.fen)

    if board.is_game_over():
        return {
            "error": "Партия окончена",
            "fen": req.fen,
        }

    # ---------------- Stockfish ----------------

    stockfish_move = _stockfish_move(board.copy())

    stockfish_result = None

    if stockfish_move is not None:
        stockfish_result = {
            "move": stockfish_move.uci(),
            "from": chess.square_name(stockfish_move.from_square),
            "to": chess.square_name(stockfish_move.to_square),
            "san": board.san(stockfish_move),
        }

        stockfish = ensure_stockfish()

        if stockfish:
            try:
                with stockfish_lock:
                    stockfish.set_fen_position(req.fen)
                    evaluation = stockfish.get_evaluation(searchtime=1000)

                stockfish_result["evaluation"] = evaluation

            except Exception as e:
                print(f"[Compare] Ошибка оценки Stockfish: {e}")

    # ---------------- Maia3 ----------------

    maia_move = _maia3_move(req, board.copy())

    maia_result = None

    if maia_move is not None:
        maia_result = {
            "move": maia_move.uci(),
            "from": chess.square_name(maia_move.from_square),
            "to": chess.square_name(maia_move.to_square),
            "san": board.san(maia_move),
            "elo": req.elo,
        }

    # ---------------- Результат ----------------

    same_move = (
        stockfish_move is not None
        and maia_move is not None
        and stockfish_move == maia_move
    )

    return {
        "fen": req.fen,
        "stockfish": stockfish_result,
        "maia3": maia_result,
        "same_move": same_move,
    }


@router.post("/api/game/finish")
def finish_game(req: FinishGameRequest, background_tasks: BackgroundTasks):
    """Сохраняет завершённую партию в БД с классификацией ходов.

    Классификация каждого хода через Stockfish (1с на ход) выполняется
    в фоне, чтобы не блокировать ответ фронтенду.
    """
    background_tasks.add_task(
        record_game,
        moves=req.moves,
        user_id=req.user_id,
        elo=req.elo,
        engine=req.engine,
        result=req.result,
        status=req.status,
    )
    return {"ok": True, "status": "scheduled"}
