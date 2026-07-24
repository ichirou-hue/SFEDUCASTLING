"""Endpoint'ы анализа: оценка Stockfish, сравнение движков, комментарий GigaChat, поиск похожих."""

import chess
from fastapi import APIRouter

from backend.api_gateway.models import FenRequest, SimilarityRequest
from backend.api_gateway.state import (
    ensure_stockfish,
    ensure_maia2,
    get_opening_info,
)
from backend.config.settings import settings   # <-- импорт настроек

router = APIRouter(tags=["analysis"])


@router.post("/api/stockfish-analyze")
def stockfish_analyze(req: FenRequest):
    """Анализирует позицию с помощью движка Stockfish."""
    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен"}

    try:
        stockfish.set_fen_position(req.fen)
        best_move = stockfish.get_best_move()
        evaluation = stockfish.get_evaluation()
        top_moves = stockfish.get_top_moves(settings.models.stockfish_top_moves)  # <-- из настроек

        return {
            "fen": req.fen,
            "best_move": best_move,
            "evaluation": evaluation,
            "top_moves": top_moves,
        }
    except Exception as e:
        print(f"[Stockfish] Ошибка: {e}")
        return {"error": str(e)}


@router.post("/api/compare-maia-stockfish")
def compare_engines(req: FenRequest):
    """Сравнивает оценку позиции от Maia2 и от Stockfish."""
    stockfish = ensure_stockfish()
    if not stockfish:
        return {"error": "Stockfish не загружен"}

    maia2_model, maia2_prepared = ensure_maia2()
    if not maia2_model or not maia2_prepared:
        return {"error": "Maia2 не загружена"}

    try:
        from maia2 import inference as maia2_inference

        stockfish.set_fen_position(req.fen)
        stockfish_best = stockfish.get_best_move()
        stockfish_eval = stockfish.get_evaluation()

        move_probs, win_prob = maia2_inference.inference_each(
            maia2_model, maia2_prepared, req.fen, req.elo, req.elo
        )
        maia_best = max(move_probs.items(), key=lambda x: x[1])[0]

        board1 = chess.Board(req.fen)
        board2 = chess.Board(req.fen)

        try:
            move1 = chess.Move.from_uci(stockfish_best)
            board1.push(move1)
            stockfish.set_fen_position(board1.fen())
            eval_after_stockfish = stockfish.get_evaluation()
        except Exception:
            eval_after_stockfish = {"type": "cp", "value": 0}

        try:
            move2 = chess.Move.from_uci(maia_best)
            board2.push(move2)
            stockfish.set_fen_position(board2.fen())
            eval_after_maia = stockfish.get_evaluation()
        except Exception:
            eval_after_maia = {"type": "cp", "value": 0}

        val1 = eval_after_stockfish.get("value", 0)
        val2 = eval_after_maia.get("value", 0)
        try:
            difference = abs(int(val1) - int(val2))
        except Exception:
            difference = 0

        return {
            "fen": req.fen,
            "stockfish": {
                "move": stockfish_best,
                "evaluation": stockfish_eval,
            },
            "maia2": {
                "move": maia_best,
                "probability": move_probs[maia_best],
                "win_probability": win_prob,
            },
            "comparison": {
                "eval_after_stockfish": eval_after_stockfish,
                "eval_after_maia": eval_after_maia,
                "difference": difference,
            },
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/api/analyze")
def analyze(req: FenRequest):
    """Получает текстовый анализ позиции от GigaChat."""
    if not settings.gigachat.auth_key:   # <-- используем настройки
        return {"message": "GigaChat API ключ не настроен.", "fen": req.fen}

    try:
        from gigachat import GigaChat

        board = chess.Board(req.fen)
        turn = "Белые" if board.turn == chess.WHITE else "Чёрные"
        move_number = board.fullmove_number

        opening = get_opening_info(req.fen)

        opening_context = ""
        if opening:
            if opening.get("name"):
                opening_context += f"\nДебют: {opening['name']}."
            if opening.get("games"):
                opening_context += "\nЗнаменитые партии с похожей позицией:"
                for g in opening["games"]:
                    opening_context += f"\n- {g}"

        prompt = (
            f"Ты — шахматный тренер-историк. Оцени позицию и дай совет простым языком.\n"
            f"Позиция (FEN): {req.fen}\n"
            f"Ход: {turn}, ход номер {move_number}.\n"
            f"{opening_context}\n\n"
            f"1. Если известен дебют — назови его и кратко объясни идею.\n"
            f"2. Если есть знаменитые партии — упомяни самую интересную (кто играл, год, чем закончилась).\n"
            f"3. Оцени кто лучше стоит, какие угрозы, что делать дальше.\n"
            f"Отвечай кратко и понятно."
        )

        with GigaChat(
            credentials=settings.gigachat.auth_key,        # <-- из настроек
            scope=settings.gigachat.scope,                 # <-- из настроек
            model=settings.gigachat.model,                 # <-- из настроек
            verify_ssl_certs=settings.gigachat.verify_ssl_certs,  # <-- из настроек
        ) as giga:
            response = giga.chat(prompt)
            message = response.choices[0].message.content

    except Exception as e:
        message = f"Ошибка GigaChat: {str(e)}"

    return {
        "message": message,
        "fen": req.fen,
    }


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