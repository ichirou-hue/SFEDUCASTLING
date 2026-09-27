import chess

from backend.llm.explainer.names import _piece_name


# ============================================================
# РАЗРЕШЁННЫЕ КООРДИНАТЫ
# ============================================================

def _line_opening_facts(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> list[dict]:
    """
    Ищет безопасные педагогические факты вида
    «ход освободил линию для слона/ладьи/ферзя».

    Условие намеренно строгое:
    - рассматривается своя дальнобойная фигура;
    - поле FROM перемещённой фигуры до хода лежало в её атаке;
    - после освобождения FROM у дальнобойной фигуры появились
      новые контролируемые поля за бывшим блокером.

    Это не стратегическая интерпретация, а геометрический факт доски.
    """
    mover = board_before.turn
    result: list[dict] = []

    for square, piece in board_before.piece_map().items():
        if piece.color != mover:
            continue

        if piece.piece_type not in (
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ):
            continue

        # Перемещённую фигуру не анализируем как «открывшуюся».
        if square == move.from_square:
            continue

        same_piece_after = board_after.piece_at(square)
        if same_piece_after != piece:
            continue

        before_attacks = set(board_before.attacks(square))
        after_attacks = set(board_after.attacks(square))

        if move.from_square not in before_attacks:
            continue

        new_squares = sorted(
            after_attacks - before_attacks,
            key=lambda sq: chess.square_name(sq),
        )

        if not new_squares:
            continue

        result.append({
            "piece": _piece_name(piece),
            "square": chess.square_name(square),
            "new_controls": [
                chess.square_name(sq)
                for sq in new_squares
            ],
        })

    return result


def _build_derived_explanation_facts(
    *,
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
) -> dict:
    is_game_over = board_after.is_game_over(claim_draw=True)
    outcome = (
        board_after.outcome(claim_draw=True)
        if is_game_over
        else None
    )

    if outcome is None:
        result = "*"
        winner = None
        termination = None
    else:
        result = board_after.result(claim_draw=True)
        winner = (
            "white"
            if outcome.winner is chess.WHITE
            else "black"
            if outcome.winner is chess.BLACK
            else None
        )
        termination = getattr(
            outcome.termination,
            "name",
            str(outcome.termination),
        )

    opponent_king = board_after.king(board_after.turn)
    opponent_king_square = (
        chess.square_name(opponent_king)
        if opponent_king is not None
        else None
    )

    return {
        "line_openings": _line_opening_facts(
            board_before,
            board_after,
            move,
        ),
        "terminal": {
            "is_game_over": is_game_over,
            "is_checkmate": board_after.is_checkmate(),
            "is_stalemate": board_after.is_stalemate(),
            "is_insufficient_material": board_after.is_insufficient_material(),
            "result": result,
            "winner": winner,
            "termination": termination,
            "opponent_king_square": opponent_king_square,
        },
    }
