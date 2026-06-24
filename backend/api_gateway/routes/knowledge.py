"""Endpoint'ы базы знаний: дебюты, проверка теории."""

import random
import chess
from fastapi import APIRouter

from backend.api_gateway.models import FenRequest
from backend.api_gateway.state import knowledge_base

router = APIRouter(tags=["knowledge"])


@router.get("/api/knowledge/openings")
def get_openings():
    """Возвращает список всех дебютов в базе знаний."""
    if not knowledge_base:
        return {"openings": [], "error": "База знаний не загружена"}
    return {"openings": knowledge_base.get("openings", [])}


@router.get("/api/knowledge/opening")
def get_opening_by_fen(fen: str = ""):
    """Ищет дебют по FEN-позиции."""
    if not knowledge_base:
        return {"error": "База знаний не загружена"}

    fen_lower = fen.lower().split()[0]

    for opening in knowledge_base.get("openings", []):
        if opening.get("fen", "").lower().startswith(fen_lower):
            return {"opening": opening}

    return {"opening": None, "message": "Дебют не найден в базе"}


@router.get("/api/knowledge/random-opening")
def get_random_opening():
    """Возвращает случайный дебют из базы знаний."""
    if not knowledge_base:
        return {"error": "База знаний не загружена"}

    opening = random.choice(knowledge_base.get("openings", []))
    return {"opening": opening}


@router.post("/api/knowledge/check-move")
def check_move(req: FenRequest):
    """Проверяет, соответствует ли текущая позиция известному дебюту."""
    if not knowledge_base:
        return {"error": "База знаний не загружена"}

    board = chess.Board(req.fen)
    current_fen = board.fen().split()[0].lower()

    for opening in knowledge_base.get("openings", []):
        opening_fen = opening.get("fen", "").split()[0].lower()
        if current_fen == opening_fen:
            return {
                "in_theory": True,
                "opening": opening.get("name"),
                "eco": opening.get("eco"),
                "description": opening.get("description"),
            }

    return {"in_theory": False, "message": "Позиция не найдена в базе теории"}
