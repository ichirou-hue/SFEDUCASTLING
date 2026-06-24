"""Игровые endpoint'ы: легальные ходы, выполнение хода, ход Maia."""

import random
import chess
from fastapi import APIRouter
from backend.api_gateway.models import FenSquare, MoveRequest, FenRequest
from backend.api_gateway.state import ensure_maia2

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
def maia_move(req: FenRequest):
    """Получает человекообразный ход от движка Maia2."""
    board = chess.Board(req.fen)
    if board.is_game_over():
        return {"error": "Партия окончена", "fen": req.fen}

    maia2_model, maia2_prepared = ensure_maia2()

    if maia2_model and maia2_prepared:
        try:
            from maia2 import inference as maia2_inference

            move_probs, win_prob = maia2_inference.inference_each(
                maia2_model, maia2_prepared, req.fen, req.elo, req.elo
            )
            if not move_probs:
                print("[Maia2] WARNING: пустой move_probs, используем случайный ход")
                move = random.choice(list(board.legal_moves))
            else:
                best_uci = max(move_probs, key=move_probs.get)
                move = chess.Move.from_uci(best_uci)
        except Exception as e:
            print(f"[Maia2] Ошибка инференса: {e}")
            import traceback
            traceback.print_exc()
            move = random.choice(list(board.legal_moves))
    else:
        print("[Maia2] ВНИМАНИЕ: maia2 не загружена, используется случайный ход")
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
