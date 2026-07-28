"""Endpoint'ы для анализа шахматной позиции: легальные ходы, ход, анализ."""

import chess
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from backend.api_gateway.state import ensure_stockfish, stockfish_lock

router = APIRouter(prefix="/api/analyze", tags=["analysis"])


# ---- Модели запросов и ответов ----
class PositionRequest(BaseModel):
    fen: str


class LegalMovesResponse(BaseModel):
    legal_moves: List[str]


class MoveRequest(BaseModel):
    fen: str
    move: str  # в формате UCI


class MoveResponse(BaseModel):
    new_fen: str
    move_san: Optional[str] = None


class AnalysisRequest(BaseModel):
    fen: str
    depth: Optional[int] = 10


class AnalysisResponse(BaseModel):
    evaluation: float
    best_move: Optional[str] = None
    depth: int


# ---- Вспомогательные функции ----
def fen_to_board(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError:
        return None


# ---- Эндпоинты ----

@router.post("/legal-moves")
async def get_legal_moves(request: PositionRequest):
    """
    Возвращает список всех легальных ходов для заданной позиции.
    """
    board = fen_to_board(request.fen)
    if board is None:
        return {"error": "Invalid FEN string", "legal_moves": []}
    moves = [move.uci() for move in board.legal_moves]
    return {"legal_moves": moves}


@router.post("/move", operation_id="analyze_make_move")
async def make_move(request: MoveRequest):
    """
    Выполняет ход и возвращает новую FEN-позицию.
    """
    board = fen_to_board(request.fen)
    if board is None:
        return {"error": "Invalid FEN string"}

    try:
        move = chess.Move.from_uci(request.move)
    except ValueError:
        return {"error": "Invalid move format (use UCI)"}

    if move not in board.legal_moves:
        return {"error": "Illegal move"}

    # Получаем SAN ДО изменения позиции
    san = board.san(move)

    board.push(move)

    return {"new_fen": board.fen(), "move_san": san}


@router.post("/position")
async def analyze_position(request: AnalysisRequest):
    """
    Упрощённый анализ позиции: оценка материала (без движка).
    """
    board = fen_to_board(request.fen)
    if board is None:
        return {"error": "Invalid FEN string"}

    # Вес фигур (в пешках)
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

    # Возвращаем оценку (положительная – белые лучше)
    return {
        "evaluation": score,
        "best_move": None,  # без движка лучший ход не определяем
        "depth": request.depth,
        "note": "Упрощённая оценка материала, Stockfish не установлен"
    }