"""Русские названия фигур и форматирование оценок. Только именительный падеж: склонение делает LLM."""
from __future__ import annotations

import chess

PIECE_RU = {
    chess.PAWN: "пешка",
    chess.KNIGHT: "конь",
    chess.BISHOP: "слон",
    chess.ROOK: "ладья",
    chess.QUEEN: "ферзь",
    chess.KING: "король",
}
_FEMININE = {chess.PAWN, chess.ROOK}

# Как в контракте 02: «белая пешка», цвет обязателен. Если canonical в dataset_tools.py
# пишет «черный» без ё — поменять здесь, иначе TQU-записи не пройдут валидатор.
BLACK_ROOT = "чёрн"


def color_adj(color: bool, piece_type: int) -> str:
    fem = piece_type in _FEMININE
    if color == chess.WHITE:
        return "белая" if fem else "белый"
    return BLACK_ROOT + ("ая" if fem else "ый")


def piece_ru(piece: chess.Piece) -> str:
    return f"{color_adj(piece.color, piece.piece_type)} {PIECE_RU[piece.piece_type]}"


def piece_sym_ru(symbol: str) -> str:
    return piece_ru(chess.Piece.from_symbol(symbol))


def side_ru(color: bool) -> str:
    return "белые" if color == chess.WHITE else BLACK_ROOT + "ые"


def side_gen_ru(color: bool) -> str:
    return "белых" if color == chess.WHITE else BLACK_ROOT + "ых"


def fmt_pawns(cp: int) -> str:
    return f"{abs(cp) / 100:.2f}"


def eval_phrase(cp: int | None, mate: int | None) -> str:
    """Оценка с точки зрения белых -> «0.45 в пользу чёрных» / «мат в 3 хода в пользу белых»."""
    if mate is not None:
        if mate == 0:
            return "мат на доске"
        side = "белых" if mate > 0 else side_gen_ru(chess.BLACK)
        return f"мат в {abs(mate)} ход{_plural(abs(mate))} в пользу {side}"
    if cp is None:
        return "нет оценки"
    if abs(cp) < 30:
        return f"примерное равенство ({fmt_pawns(cp)})"
    side = "белых" if cp > 0 else side_gen_ru(chess.BLACK)
    return f"{fmt_pawns(cp)} в пользу {side}"


def eval_words(cp: int | None, mate: int | None) -> str:
    """Оценка с точки зрения белых словами, без чисел: для текста, который читает человек.
    Пороги — как у шкалы шансов на выигрыш lichess: 0.3 / 1 / 3 пешки."""
    if mate is not None:
        if mate == 0:
            return "мат на доске"
        side = "белых" if mate > 0 else side_gen_ru(chess.BLACK)
        return f"у {side} форсированный мат в {abs(mate)} ход{_plural(abs(mate))}"
    if cp is None:
        return "оценки нет"
    a = abs(cp)
    if a < 30:
        return "примерно равная позиция"
    side = "белых" if cp > 0 else side_gen_ru(chess.BLACK)
    if a < 100:
        return f"небольшой перевес {side}"
    if a < 300:
        return f"заметный перевес {side}"
    return f"решающий перевес {side}"


def _plural(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return ""
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "а"
    return "ов"
