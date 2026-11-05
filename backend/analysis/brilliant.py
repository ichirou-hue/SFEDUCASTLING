"""Deterministic brilliant-move heuristic for Maia game achievements.

A move is not called brilliant just because Stockfish likes it.  It must be
Stockfish's best move and deliberately leave a valuable moved piece available
to a legal recapture while giving up at least two points of net material.
This keeps ``brilliant`` orthogonal to the ordinary best/excellent/good move
classification used by accuracy statistics.
"""

from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
MIN_NET_SACRIFICE = 2


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return PIECE_VALUES.get(captured.piece_type, 0) if captured else 0


def brilliant_sacrifice(
    board: chess.Board,
    move: chess.Move,
    *,
    best_uci: str | None,
    classification: str | None,
) -> dict:
    """Return brilliant metadata for one already engine-scored player move.

    The heuristic intentionally stays conservative:
    * the played UCI move must equal Stockfish's best move;
    * the normal quality class must be best/excellent;
    * the position must offer more than one legal move (not forced);
    * after the move the opponent must have a legal capture of the moved piece;
    * the offered piece minus material captured by the move must be worth >= 2.
    """
    result = {
        "is_brilliant": False,
        "sacrifice_value": 0,
        "reason": None,
    }
    if not best_uci or best_uci != move.uci():
        return result
    if classification not in {"best", "excellent"}:
        return result
    if move not in board.legal_moves:
        return result
    if sum(1 for _ in board.legal_moves) <= 1:
        return result

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None or moving_piece.piece_type == chess.KING:
        return result

    captured_value = _captured_value(board, move)
    after = board.copy(stack=False)
    after.push(move)
    offered_piece = after.piece_at(move.to_square)
    if offered_piece is None or offered_piece.color != moving_piece.color:
        return result

    offered_value = PIECE_VALUES.get(offered_piece.piece_type, 0)
    net_sacrifice = offered_value - captured_value
    if net_sacrifice < MIN_NET_SACRIFICE:
        return result

    can_capture_offered_piece = False
    for reply in after.legal_moves:
        if reply.to_square != move.to_square or not after.is_capture(reply):
            continue
        victim = after.piece_at(reply.to_square)
        if victim is not None and victim.color == moving_piece.color:
            can_capture_offered_piece = True
            break

    if not can_capture_offered_piece:
        return result

    return {
        "is_brilliant": True,
        "sacrifice_value": int(net_sacrifice),
        "reason": "best_material_sacrifice",
    }
