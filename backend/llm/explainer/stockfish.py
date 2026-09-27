import chess

from backend.api_gateway.state import ensure_stockfish, reset_stockfish, stockfish_lock
from backend.llm.explainer.move_facts import _move_info


# ============================================================
# STOCKFISH
# ============================================================

def _normalise_stockfish_result(
    board: chess.Board,
    infos: list,
) -> dict:

    if not infos:

        return {
            "available": True,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],
        }

    top_moves = []

    for info in infos:

        if not isinstance(info, dict):
            continue

        move_uci = info.get("Move")

        if not move_uci:
            continue

        try:

            move = chess.Move.from_uci(
                move_uci
            )

        except ValueError:

            continue

        if move not in board.legal_moves:
            continue

        move_info = _move_info(
            board,
            move,
        )

        if move_info is None:
            continue

        evaluation = info.get(
            "Centipawn"
        )

        mate = info.get(
            "Mate"
        )

        depth = (
            info.get("Depth")
            or info.get("SelectiveDepth")
        )

        top_moves.append({
            "move": move_info,

            "evaluation": evaluation,

            "mate": mate,

            "depth": depth,
        })

    if not top_moves:

        return {
            "available": True,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],
        }

    best = top_moves[0]

    if best.get("mate") is not None:

        evaluation = {
            "type": "mate",

            "value": best["mate"],
        }

    elif best.get("evaluation") is not None:

        evaluation = {
            "type": "cp",

            "value": best["evaluation"],
        }

    else:

        evaluation = None

    return {
        "available": True,

        "best_move": best["move"],

        "evaluation": evaluation,

        "top_moves": top_moves,
    }


def _stockfish_analysis(
    board: chess.Board,
) -> dict:

    stockfish = ensure_stockfish()

    if stockfish is None:

        return {
            "available": False,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],

            "error": "Stockfish недоступен.",
        }

    try:

        with stockfish_lock:

            stockfish.set_fen_position(
                board.fen()
            )

            infos = stockfish.get_top_moves(
                5
            )

        result = _normalise_stockfish_result(
            board,
            infos,
        )

        if result.get("best_move"):

            return result

        print(
            "[ChessExplainer] "
            "Stockfish MultiPV не вернул "
            "лучший ход. Переходим к get_best_move()."
        )

    except Exception as e:

        print(
            "[ChessExplainer] "
            f"Stockfish MultiPV error: {e}"
        )

    try:

        with stockfish_lock:

            stockfish.set_fen_position(
                board.fen()
            )

            move_uci = (
                stockfish.get_best_move()
            )

        if not move_uci:

            raise RuntimeError(
                "Stockfish не вернул bestmove."
            )

        move = chess.Move.from_uci(
            move_uci
        )

        if move not in board.legal_moves:

            raise RuntimeError(
                "Stockfish вернул "
                f"нелегальный ход: {move_uci}"
            )

        move_info = _move_info(
            board,
            move,
        )

        if move_info is None:

            raise RuntimeError(
                "Не удалось получить информацию "
                "о bestmove."
            )

        return {
            "available": True,

            "best_move": move_info,

            "evaluation": None,

            "top_moves": [
                {
                    "move": move_info,

                    "evaluation": None,

                    "mate": None,

                    "depth": None,
                }
            ],
        }

    except Exception as e:

        print(
            "[ChessExplainer] "
            f"Stockfish process error: {e}"
        )

        try:

            reset_stockfish()

        except Exception as reset_error:

            print(
                "[ChessExplainer] "
                f"Stockfish reset error: {reset_error}"
            )

        return {
            "available": False,

            "best_move": None,

            "evaluation": None,

            "top_moves": [],

            "error": str(e),
        }
