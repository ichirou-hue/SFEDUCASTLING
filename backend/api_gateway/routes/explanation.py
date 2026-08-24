from fastapi import APIRouter, HTTPException

from backend.api_gateway.models import ExplainMoveRequest
from backend.llm.chess_explainer import explain_move


router = APIRouter(tags=["explanation"])


@router.post("/api/explain-move")
def explain_move_endpoint(req: ExplainMoveRequest):
    """Объясняет последний ход игрока через hybrid-анализ."""

    try:
        chess_request = type(
            "FenRequestAdapter",
            (),
            {
                "fen": req.fen,
                "elo": req.elo,
                "moves": req.moves,
            },
        )()

        return explain_move(
            req=chess_request,
            played_move=req.move,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        print(f"[Explanation] Ошибка: {e}")

        raise HTTPException(
            status_code=500,
            detail="Ошибка hybrid-анализа",
        )