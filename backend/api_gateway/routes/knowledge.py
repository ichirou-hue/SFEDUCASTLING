"""Endpoint'ы базы знаний: дебюты, проверка теории."""

import random
import chess
from fastapi import APIRouter

from backend.api_gateway.models import FenRequest
from backend.api_gateway import state

router = APIRouter(tags=["knowledge"])


def _pieces(fen: str) -> str:
    return fen.strip().lower().split()[0]


def _covering_openings(current: str) -> list[dict]:
    """Дебюты, внутри которых встречается текущая раскладка фигур."""
    result = []
    for opening in state.knowledge_base.get("openings", []):
        fens_low = {f.lower() for f in opening.get("fens", [])}
        if fens_low and current in fens_low:
            result.append(opening)
        elif not fens_low and opening.get("fen", "").split()[0].lower() == current:
            result.append(opening)
    result.sort(key=lambda o: len(o.get("moves", [])))
    return result


def _book_move(opening: dict, current: str) -> str | None:
    fens = opening.get("fens")
    moves = opening.get("moves")
    if not fens or not moves:
        return None
    fen_index = {f.lower(): i for i, f in enumerate(fens)}.get(current)
    if fen_index is None:
        return None
    if fen_index < len(moves):
        return moves[fen_index]
    return None


@router.get("/api/knowledge/openings")
def get_openings():
    """Возвращает список всех дебютов в базе знаний."""
    if not state.knowledge_base:
        return {"openings": [], "error": "База знаний не загружена"}
    return {"openings": state.knowledge_base.get("openings", [])}


@router.get("/api/knowledge/opening")
def get_opening_by_fen(fen: str = ""):
    """Ищет самый глубокий дебют, покрывающий заданную позицию."""
    if not state.knowledge_base:
        return {"error": "База знаний не загружена"}

    covering = _covering_openings(_pieces(fen))
    if not covering:
        return {"opening": None, "message": "Дебют не найден в базе"}

    return {"opening": covering[-1]}


@router.get("/api/knowledge/random-opening")
def get_random_opening():
    """Возвращает случайный дебют из базы знаний."""
    if not state.knowledge_base:
        return {"error": "База знаний не загружена"}

    opening = random.choice(state.knowledge_base.get("openings", []))
    return {"opening": opening}


@router.post("/api/knowledge/check-move")
def check_move(req: FenRequest):
    """Проверяет, соответствует ли текущая позиция известному дебюту."""
    if not state.knowledge_base:
        return {"error": "База знаний не загружена"}

    board = chess.Board(req.fen)
    current = board.fen().split()[0].lower()
    covering = _covering_openings(current)
    if not covering:
        return {"in_theory": False, "message": "Позиция не найдена в базе теории"}

    opening = covering[-1]
    book = []
    for candidate in covering:
        move = _book_move(candidate, current)
        if move and move not in book:
            book.append(move)

    return {
        "in_theory": True,
        "opening": opening.get("name"),
        "eco": opening.get("eco"),
        "pgn": opening.get("pgn"),
        "next_moves": book,
    }
