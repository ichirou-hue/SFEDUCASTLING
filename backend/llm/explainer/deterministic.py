import chess

from backend.llm.explainer.move_facts import _join_squares
from backend.llm.explainer.names import (
    _piece_name_accusative,
    _piece_name_capitalized,
    _piece_word_accusative,
)


# ============================================================
# ДЕТЕРМИНИРОВАННОЕ ОБЪЯСНЕНИЕ
# ============================================================

def _deterministic_explanation(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    facts: dict,
) -> str:

    piece = board_before.piece_at(
        move.from_square
    )

    if piece is None:

        return (
            "Stockfish рекомендует этот ход. "
            "Фигура перемещается на указанное поле. "
            "Точные данные хода получены из позиции."
        )

    san = board_before.san(
        move
    )

    quality_text = (
        f"Stockfish рекомендует ход {san}. "
    )

    from_square = chess.square_name(
        move.from_square
    )

    to_square = chess.square_name(
        move.to_square
    )

    piece_text = _piece_name_capitalized(
        piece
    )

    # --------------------------------------------------------
    # ПРЕВРАЩЕНИЕ — обрабатываем раньше шаха/взятия, потому что
    # превращение может одновременно быть взятием, шахом или матом.
    # --------------------------------------------------------

    if facts.get("is_promotion"):

        promotion_piece = str(
            facts.get("promotion_piece") or "фигура"
        ).lower()
        promotion_piece_acc = _piece_word_accusative(
            promotion_piece
        )

        movement_text = (
            f"Пешка переходит с {from_square} на {to_square}"
        )

        if facts.get("is_capture"):
            captured = str(
                facts.get("captured_piece") or "фигура соперника"
            )
            captured_acc = _piece_name_accusative(captured)
            movement_text += (
                f", берёт {captured_acc} на {to_square}"
            )

        movement_text += (
            f" и превращается в {promotion_piece_acc}. "
        )

        opponent_king = board_after.king(board_after.turn)
        opponent_king_square = (
            chess.square_name(opponent_king)
            if opponent_king is not None
            else None
        )

        if facts.get("is_checkmate"):
            if opponent_king_square:
                movement_text += (
                    f"После превращения {promotion_piece} на {to_square} "
                    f"ставит мат королю соперника на "
                    f"{opponent_king_square}."
                )
            else:
                movement_text += (
                    f"После превращения {promotion_piece} ставит мат."
                )
        elif facts.get("is_check"):
            if opponent_king_square:
                movement_text += (
                    f"После превращения {promotion_piece} на {to_square} "
                    f"объявляет шах королю соперника на "
                    f"{opponent_king_square}."
                )
            else:
                movement_text += (
                    f"После превращения {promotion_piece} объявляет шах."
                )

        return quality_text + movement_text

    # --------------------------------------------------------
    # МАТ
    # --------------------------------------------------------

    if facts.get(
        "is_checkmate"
    ):

        return (
            quality_text
            + f"{piece_text} переходит с "
            f"{from_square} на {to_square} "
            f"и ставит мат. "
            f"После этого у соперника нет "
            f"легального ответа."
        )

    # --------------------------------------------------------
    # ВЗЯТИЕ НА ПРОХОДЕ
    # --------------------------------------------------------

    if facts.get("is_en_passant"):

        captured_square = str(
            facts.get("captured_square") or ""
        )
        mover_color = str(
            facts.get("piece_color") or ""
        ).lower()
        mover_side = "белая" if mover_color == "white" else "чёрная"
        captured_side = "чёрную" if mover_color == "white" else "белую"

        return (
            quality_text
            + f"{mover_side.capitalize()} пешка выполняет взятие на проходе: "
            f"переходит с {from_square} на {to_square} и снимает "
            f"{captured_side} пешку с {captured_square}."
        )

    # --------------------------------------------------------
    # ВЗЯТИЕ
    # --------------------------------------------------------

    if facts.get(
        "is_capture"
    ):

        captured = facts.get(
            "captured_piece"
        )

        if captured:

            captured_square = (
                facts.get(
                    "captured_square"
                )
                or to_square
            )

            captured_acc = _piece_name_accusative(
                str(captured)
            )

            capture_geometry = ""

            if (
                facts.get("movement_type") == "diagonal"
                and piece.piece_type in {
                    chess.BISHOP,
                    chess.QUEEN,
                }
            ):
                capture_geometry = " по диагонали"

            elif (
                facts.get("movement_type") == "horizontal"
                and piece.piece_type in {
                    chess.ROOK,
                    chess.QUEEN,
                }
            ):
                capture_geometry = " по горизонтали"

            elif (
                facts.get("movement_type") == "vertical"
                and piece.piece_type in {
                    chess.ROOK,
                    chess.QUEEN,
                }
            ):
                capture_geometry = " по вертикали"

            elif (
                facts.get("movement_type") == "knight_jump"
                and piece.piece_type == chess.KNIGHT
            ):
                capture_geometry = " ходом буквой «Г»"

            elif (
                facts.get("movement_type") == "king_step"
                and piece.piece_type == chess.KING
            ):
                capture_geometry = " на соседнюю клетку"

            elif (
                facts.get("movement_type") == "pawn"
                and piece.piece_type == chess.PAWN
                and facts.get("is_capture")
                and not facts.get("is_en_passant")
                and not facts.get("is_promotion")
            ):
                capture_geometry = " по диагонали вперёд на одну клетку"

            elif (
                facts.get("movement_type") == "pawn"
                and piece.piece_type == chess.PAWN
                and not facts.get("is_capture")
                and not facts.get("is_en_passant")
                and not facts.get("is_promotion")
                and not facts.get("pawn_double_step")
            ):
                capture_geometry = " прямо вперёд на одну клетку"

            return (
                quality_text
                + f"{piece_text} переходит с "
                f"{from_square} на {to_square}"
                f"{capture_geometry} и берёт "
                f"{captured_acc} на "
                f"{captured_square}. "
                f"После хода взятая фигура "
                f"удалена с доски."
            )

    # --------------------------------------------------------
    # ШАХ
    # --------------------------------------------------------

    if facts.get(
        "is_check"
    ):

        return (
            quality_text
            + f"{piece_text} переходит с "
            f"{from_square} на {to_square} "
            f"и даёт шах. "
            f"Король соперника находится "
            f"под атакой."
        )

    # --------------------------------------------------------
    # РОКИРОВКА
    # --------------------------------------------------------

    if facts.get(
        "is_castling"
    ):

        is_kingside = (
            chess.square_file(move.to_square)
            > chess.square_file(move.from_square)
        )
        side = "короткая" if is_kingside else "длинная"

        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(
            7 if is_kingside else 0,
            rank,
        )
        rook_to = chess.square(
            5 if is_kingside else 3,
            rank,
        )

        rook_from_name = chess.square_name(rook_from)
        rook_to_name = chess.square_name(rook_to)

        return (
            quality_text
            + f"Этим ходом выполняется {side} рокировка: "
            f"король переходит с {from_square} на {to_square}, "
            f"а ладья одновременно переходит с "
            f"{rook_from_name} на {rook_to_name}."
        )

    # --------------------------------------------------------
    # ПЕШКА
    # --------------------------------------------------------

    if piece.piece_type == chess.PAWN:

        if facts.get(
            "pawn_double_step"
        ):

            movement_text = (
                f"Пешка переходит с "
                f"{from_square} на "
                f"{to_square} двойным шагом."
            )

        else:

            movement_text = (
                f"Пешка переходит с "
                f"{from_square} на "
                f"{to_square} прямо вперёд "
                "на одну клетку."
            )

        if facts.get(
            "pawn_reaches_center"
        ):

            movement_text += (
                f" Она занимает "
                f"центральное поле "
                f"{to_square}."
            )

        new_controls = facts.get(
            "pawn_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + movement_text
                + " После хода пешка "
                "контролирует "
                + _join_squares(
                    new_controls
                )
                + "."
            )

        return (
            quality_text
            + movement_text
            + " После хода меняется "
            "набор полей, которые "
            "контролирует пешка."
        )

    # --------------------------------------------------------
    # КОНЬ
    # --------------------------------------------------------

    if piece.piece_type == chess.KNIGHT:

        new_controls = facts.get(
            "knight_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} прыгает "
                f"с {from_square} на "
                f"{to_square}. "
                f"После хода конь получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} прыгает "
            f"с {from_square} на "
            f"{to_square}. "
            f"После хода меняется набор "
            f"полей, контролируемых конём."
        )

    # --------------------------------------------------------
    # СЛОН
    # --------------------------------------------------------

    if piece.piece_type == chess.BISHOP:

        new_controls = facts.get(
            "bishop_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} перемещается "
                f"по диагонали с "
                f"{from_square} на "
                f"{to_square}. "
                f"После хода слон получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} перемещается "
            f"по диагонали с "
            f"{from_square} на "
            f"{to_square}. "
            f"После хода меняется набор "
            f"полей, контролируемых слоном."
        )

    # --------------------------------------------------------
    # ЛАДЬЯ
    # --------------------------------------------------------

    if piece.piece_type == chess.ROOK:

        direction = (
            "по вертикали"
            if facts.get(
                "movement_type"
            ) == "vertical"
            else "по горизонтали"
        )

        new_controls = facts.get(
            "rook_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} перемещается "
                f"с {from_square} на "
                f"{to_square} {direction}. "
                f"После хода ладья получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} перемещается "
            f"с {from_square} на "
            f"{to_square} {direction}. "
            f"После хода меняется набор "
            f"полей, контролируемых ладьёй."
        )

    # --------------------------------------------------------
    # ФЕРЗЬ
    # --------------------------------------------------------

    if piece.piece_type == chess.QUEEN:

        movement_type = facts.get(
            "movement_type"
        )

        if movement_type == "diagonal":

            direction = "по диагонали"

        elif movement_type == "vertical":

            direction = "по вертикали"

        else:

            direction = "по горизонтали"

        new_controls = facts.get(
            "queen_new_controls",
            [],
        )

        if new_controls:

            return (
                quality_text
                + f"{piece_text} перемещается "
                f"с {from_square} на "
                f"{to_square} {direction}. "
                f"После хода ферзь получает "
                f"контроль над "
                f"{_join_squares(new_controls)}."
            )

        return (
            quality_text
            + f"{piece_text} перемещается "
            f"с {from_square} на "
            f"{to_square} {direction}. "
            f"После хода меняется набор "
            f"полей, контролируемых ферзём."
        )

    # --------------------------------------------------------
    # КОРОЛЬ
    # --------------------------------------------------------

    if piece.piece_type == chess.KING:

        return (
            quality_text
            + f"{piece_text} переходит с "
            f"{from_square} на "
            f"{to_square}. "
            f"После хода меняется набор "
            f"полей, контролируемых королём."
        )

    return (
        quality_text
        + f"{piece_text} переходит с "
        f"{from_square} на "
        f"{to_square}. "
        f"Это конкретное перемещение "
        f"Stockfish рекомендует "
        f"в данной позиции."
    )
