"""Игровые endpoint'ы: легальные ходы, выполнение хода, ход AI."""

import random
import chess
from fastapi import APIRouter
from backend.api_gateway.models import FenSquare, MoveRequest, FenRequest
from backend.api_gateway.state import ensure_stockfish, reset_stockfish, stockfish_lock

router = APIRouter(tags=["game"])


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


@router.post("/api/maia-move")
@router.post("/api/stockfish-move")
def ai_move(req: FenRequest):
    """Получает ход от Stockfish."""
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Партия окончена", "fen": req.fen}

    stockfish = ensure_stockfish()
    if stockfish:
        try:
            with stockfish_lock:
                print(f"[AI] Начало расчёта хода...")
                stockfish.set_fen_position(req.fen)
                best_uci = stockfish.get_best_move_time(10000)
                print(f"[AI] Ход: {best_uci}")
            if best_uci:
                move = chess.Move.from_uci(best_uci)
            else:
                print("[AI] Ход не получен — случайный")
                move = random.choice(list(board.legal_moves))
        except Exception as e:
            print(f"[Stockfish] Ошибка хода: {e}")
            reset_stockfish()
            move = random.choice(list(board.legal_moves))
    else:
        print("[AI] Stockfish недоступен, случайный ход")
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

    return {
        "fen": board.fen(),
        "san": san,
        "from": from_name,
        "to": to_name,
        "status": status,
        "turn": "w" if board.turn == chess.WHITE else "b",
    }
