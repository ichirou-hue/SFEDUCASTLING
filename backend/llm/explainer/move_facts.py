import chess

from backend.llm.explainer.names import PIECE_NAMES, _piece_name


# ============================================================
# ИНФОРМАЦИЯ О ХОДЕ
# ============================================================

def _move_info(
    board: chess.Board,
    move: chess.Move | None,
) -> dict | None:

    if move is None:
        return None

    if move not in board.legal_moves:
        return None

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    return {
        "uci": move.uci(),
        "san": board.san(move),
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "piece": _piece_name(piece),
    }


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _join_squares(squares: list[str]) -> str:

    if not squares:
        return ""

    if len(squares) == 1:
        return squares[0]

    if len(squares) == 2:
        return f"{squares[0]} и {squares[1]}"

    return ", ".join(squares[:-1]) + " и " + squares[-1]


def _piece_movement_type(
    board: chess.Board,
    move: chess.Move,
) -> str | None:

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    piece_type = piece.piece_type

    if piece_type == chess.PAWN:
        return "pawn"

    if piece_type == chess.KNIGHT:
        return "knight_jump"

    if piece_type == chess.BISHOP:
        return "diagonal"

    if piece_type == chess.ROOK:

        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)

        if from_file == to_file:
            return "vertical"

        return "horizontal"

    if piece_type == chess.QUEEN:

        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)

        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)

        if from_file == to_file:
            return "vertical"

        if from_rank == to_rank:
            return "horizontal"

        return "diagonal"

    if piece_type == chess.KING:

        if board.is_castling(move):
            return "castling"

        return "king_step"

    return None


# ============================================================
# ФАКТЫ ХОДА
# ============================================================

def _move_facts(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> dict:

    piece = board_before.piece_at(move.from_square)

    if piece is None:
        raise ValueError(
            f"На поле {chess.square_name(move.from_square)} "
            f"нет фигуры."
        )

    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)

    center = {
        chess.D4,
        chess.E4,
        chess.D5,
        chess.E5,
    }

    # --------------------------------------------------------
    # ВЗЯТИЕ
    # --------------------------------------------------------

    captured_piece = None
    captured_square = None

    if board_before.is_en_passant(move):

        captured_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square),
        )

        captured_piece = board_before.piece_at(
            captured_square
        )

    else:

        captured_piece = board_before.piece_at(
            move.to_square
        )

        if captured_piece is not None:
            captured_square = move.to_square

    # --------------------------------------------------------
    # КОНТРОЛЬ ПОЛЕЙ
    # --------------------------------------------------------

    controls_before_set = set(
        board_before.attacks(move.from_square)
    )

    controls_after_set = set(
        board_after.attacks(move.to_square)
    )

    controls_before = sorted(
        chess.square_name(square)
        for square in controls_before_set
    )

    controls_after = sorted(
        chess.square_name(square)
        for square in controls_after_set
    )

    new_controls = sorted(
        chess.square_name(square)
        for square in (
            controls_after_set - controls_before_set
        )
    )

    lost_controls = sorted(
        chess.square_name(square)
        for square in (
            controls_before_set - controls_after_set
        )
    )

    controls_center_before = sorted(
        chess.square_name(square)
        for square in (
            controls_before_set & center
        )
    )

    controls_center_after = sorted(
        chess.square_name(square)
        for square in (
            controls_after_set & center
        )
    )

    new_center_controls = sorted(
        chess.square_name(square)
        for square in (
            (controls_after_set - controls_before_set) & center
        )
    )

    # --------------------------------------------------------
    # БАЗОВЫЕ ФАКТЫ
    # --------------------------------------------------------

    facts = {
        "piece": _piece_name(piece),

        "piece_type": piece.piece_type,

        "piece_symbol": piece.symbol().lower(),

        "piece_color": (
            "white"
            if piece.color == chess.WHITE
            else "black"
        ),

        "movement_type": _piece_movement_type(
            board_before,
            move,
        ),

        "from": from_square,

        "to": to_square,

        "captured_piece": (
            _piece_name(captured_piece)
            if captured_piece
            else None
        ),

        "captured_square": (
            chess.square_name(captured_square)
            if captured_square is not None
            else None
        ),

        "is_capture": board_before.is_capture(move),

        "is_en_passant": board_before.is_en_passant(move),

        "is_check": board_after.is_check(),

        "is_checkmate": board_after.is_checkmate(),

        "is_castling": board_before.is_castling(move),

        "is_promotion": move.promotion is not None,

        "promotion_piece": (
            PIECE_NAMES.get(
                move.promotion,
                chess.piece_name(move.promotion),
            )
            if move.promotion
            else None
        ),

        "turn": (
            "white"
            if board_before.turn == chess.WHITE
            else "black"
        ),

        "from_file": chess.square_file(move.from_square),

        "from_rank": chess.square_rank(move.from_square),

        "to_file": chess.square_file(move.to_square),

        "to_rank": chess.square_rank(move.to_square),

        "moves_to_center": move.to_square in center,

        "from_center": move.from_square in center,

        "occupies_center": move.to_square in center,

        "controls_before": controls_before,

        "controls_after": controls_after,

        "new_controls": new_controls,

        "lost_controls": lost_controls,

        "controls_center_before": controls_center_before,

        "controls_center_after": controls_center_after,

        "new_center_controls": new_center_controls,

        "gives_check": board_after.is_check(),
    }

    # --------------------------------------------------------
    # ПЕШКА
    # --------------------------------------------------------

    if piece.piece_type == chess.PAWN:

        rank_diff = abs(
            chess.square_rank(move.to_square)
            - chess.square_rank(move.from_square)
        )

        facts.update({
            "pawn_double_step": rank_diff == 2,

            "pawn_controls": controls_after,

            "pawn_controls_before": controls_before,

            "pawn_controls_after": controls_after,

            "pawn_new_controls": new_controls,

            "pawn_lost_controls": lost_controls,

            "pawn_controls_center": controls_center_after,

            "pawn_new_center_controls": new_center_controls,

            "pawn_reaches_center": move.to_square in center,
        })

    # --------------------------------------------------------
    # КОНЬ
    # --------------------------------------------------------

    elif piece.piece_type == chess.KNIGHT:

        starting_squares = {
            chess.G1,
            chess.B1,
            chess.G8,
            chess.B8,
        }

        facts.update({
            "knight_controls_before": controls_before,

            "knight_controls_after": controls_after,

            "knight_new_controls": new_controls,

            "knight_lost_controls": lost_controls,

            "knight_controls_center": controls_center_after,

            "knight_new_center_controls": new_center_controls,

            "knight_develops": (
                move.from_square in starting_squares
                and move.to_square not in starting_squares
            ),
        })

    # --------------------------------------------------------
    # СЛОН
    # --------------------------------------------------------

    elif piece.piece_type == chess.BISHOP:

        facts.update({
            "bishop_controls_before": controls_before,

            "bishop_controls_after": controls_after,

            "bishop_new_controls": new_controls,

            "bishop_lost_controls": lost_controls,

            "bishop_controls_center": controls_center_after,

            "bishop_new_center_controls": new_center_controls,

            "is_diagonal_move": True,
        })

    # --------------------------------------------------------
    # ЛАДЬЯ
    # --------------------------------------------------------

    elif piece.piece_type == chess.ROOK:

        facts.update({
            "rook_controls_before": controls_before,

            "rook_controls_after": controls_after,

            "rook_new_controls": new_controls,

            "rook_lost_controls": lost_controls,

            "rook_controls_center": controls_center_after,

            "rook_new_center_controls": new_center_controls,

            "is_file_move": (
                chess.square_file(move.from_square)
                == chess.square_file(move.to_square)
            ),

            "is_rank_move": (
                chess.square_rank(move.from_square)
                == chess.square_rank(move.to_square)
            ),
        })

    # --------------------------------------------------------
    # ФЕРЗЬ
    # --------------------------------------------------------

    elif piece.piece_type == chess.QUEEN:

        facts.update({
            "queen_controls_before": controls_before,

            "queen_controls_after": controls_after,

            "queen_new_controls": new_controls,

            "queen_lost_controls": lost_controls,

            "queen_controls_center": controls_center_after,

            "queen_new_center_controls": new_center_controls,
        })

    # --------------------------------------------------------
    # КОРОЛЬ
    # --------------------------------------------------------

    elif piece.piece_type == chess.KING:

        facts.update({
            "king_controls_before": controls_before,

            "king_controls_after": controls_after,

            "king_new_controls": new_controls,

            "king_lost_controls": lost_controls,
        })

    return facts
