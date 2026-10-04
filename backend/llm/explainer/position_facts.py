import chess

from backend.llm.explainer.names import PIECE_VALUES, _piece_name


# ============================================================
# ФАКТЫ ИЗМЕНЕНИЯ ПОЗИЦИИ
# ============================================================

def _position_change_facts(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> dict:

    mover = board_before.turn
    opponent = not mover

    attacked_before = set()
    attacked_after = set()

    for square, piece in board_before.piece_map().items():

        if piece.color != mover:
            continue

        attacked_before.update(
            board_before.attacks(square)
        )

    for square, piece in board_after.piece_map().items():

        if piece.color != mover:
            continue

        attacked_after.update(
            board_after.attacks(square)
        )

    newly_attacked_pieces = []

    for square, piece in board_after.piece_map().items():

        if piece.color != opponent:
            continue

        if (
            square in attacked_after
            and square not in attacked_before
        ):

            newly_attacked_pieces.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),

                "piece_type": piece.piece_type,

                "value": PIECE_VALUES.get(
                    piece.piece_type,
                    0,
                ),
            })

    no_longer_attacked = []

    for square, piece in board_after.piece_map().items():

        if piece.color != opponent:
            continue

        if (
            square in attacked_before
            and square not in attacked_after
        ):

            no_longer_attacked.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),
            })

    newly_defended = []

    for square, piece in board_after.piece_map().items():

        if piece.color != mover:
            continue

        defenders_before = len(
            board_before.attackers(
                mover,
                square,
            )
        )

        defenders_after = len(
            board_after.attackers(
                mover,
                square,
            )
        )

        if defenders_after > defenders_before:

            newly_defended.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),
            })

    opponent_king = board_after.king(opponent)

    king_attackers = []

    if opponent_king is not None:

        king_attackers = [
            chess.square_name(square)
            for square in board_after.attackers(
                mover,
                opponent_king,
            )
        ]

    pinned_pieces = []

    for square, piece in board_after.piece_map().items():

        if piece.color != opponent:
            continue

        if board_after.is_pinned(
            opponent,
            square,
        ):

            pinned_pieces.append({
                "square": chess.square_name(square),

                "piece": _piece_name(piece),
            })

    legal_replies = list(
        board_after.legal_moves
    )

    return {
        "newly_attacked_pieces": newly_attacked_pieces,

        "no_longer_attacked": no_longer_attacked,

        "newly_defended": newly_defended,

        "opponent_king_attackers": king_attackers,

        "pinned_pieces": pinned_pieces,

        "opponent_legal_moves": len(
            legal_replies
        ),

        "opponent_in_check": (
            board_after.is_check()
        ),
    }
