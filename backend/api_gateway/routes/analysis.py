"""Endpoint'ы анализа: оценка Stockfish, поиск похожих позиций."""

import chess
from fastapi import APIRouter

from backend.api_gateway.models import FenRequest, SimilarityRequest
from backend.api_gateway.state import (
    ensure_stockfish,
    reset_stockfish,
    stockfish_lock,
)

router = APIRouter(tags=["analysis"])


@router.post("/api/analyze")
@router.post("/api/stockfish-analyze")
def stockfish_analyze(req: FenRequest):
    """Анализирует позицию с помощью движка Stockfish."""
    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен"}

    try:
        with stockfish_lock:
            print("[Analysis] Начало анализа позиции...")
            stockfish.set_fen_position(req.fen)
            stockfish.update_engine_parameters({"UCI_LimitStrength": False})
            best_move = stockfish.get_best_move_time(10000)
            evaluation = stockfish.get_evaluation()
            top_moves = stockfish.get_top_moves(5)
            print(f"[Analysis] Лучший ход: {best_move}, оценка: {evaluation}")

        return {
            "fen": req.fen,
            "best_move": best_move,
            "evaluation": evaluation,
            "top_moves": top_moves,
        }
    except Exception as e:
        print(f"[Stockfish] Ошибка: {e}")
        reset_stockfish()
        return {"error": str(e)}


@router.post("/api/similarity/search")
def similarity_search(req: SimilarityRequest):
    """Ищет позиции, похожие на заданную FEN, в векторной базе данных."""
    try:
        from backend.perception.embedder.model import get_embedding
        from backend.perception.embedder.vector_store import search_similar

        embedding = get_embedding(req.fen)
        results = search_similar(embedding, top_k=req.top_k)
        return {
            "fen": req.fen,
            "top_k": req.top_k,
            "results": results,
            "count": len(results),
        }
    except ImportError:
        return {
            "fen": req.fen,
            "top_k": req.top_k,
            "error": "Vector search modules not available",
            "results": [],
        }
    except Exception as e:
        return {"error": str(e), "fen": req.fen}
