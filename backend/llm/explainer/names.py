import re

import chess

from backend.llm.explainer import constants as _constants  # noqa: F401


# ============================================================
# НАЗВАНИЯ
# ============================================================

PIECE_NAMES = {
    chess.PAWN: "пешка",
    chess.KNIGHT: "конь",
    chess.BISHOP: "слон",
    chess.ROOK: "ладья",
    chess.QUEEN: "ферзь",
    chess.KING: "король",
}

COLOR_NAMES = {
    chess.WHITE: "белых",
    chess.BLACK: "чёрных",
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _piece_name(piece: chess.Piece | None) -> str:
    if piece is None:
        return "нет фигуры"

    return (
        f"{PIECE_NAMES[piece.piece_type]} "
        f"{COLOR_NAMES[piece.color]}"
    )


def _piece_name_capitalized(piece: chess.Piece | None) -> str:
    text = _piece_name(piece)

    if not text:
        return text

    return text[0].upper() + text[1:]


def _piece_word_accusative(piece_word: str) -> str:
    forms = {
        "пешка": "пешку",
        "конь": "коня",
        "слон": "слона",
        "ладья": "ладью",
        "ферзь": "ферзя",
        "король": "короля",
    }
    return forms.get(piece_word, piece_word)


def _piece_name_accusative(piece_text: str) -> str:
    """
    "ферзь чёрных" -> "ферзя чёрных"
    "пешка белых" -> "пешку белых"
    """
    lowered = piece_text.lower().strip()
    for canonical in PIECE_NAMES.values():
        if lowered.startswith(canonical):
            suffix = piece_text[len(canonical):]
            return _piece_word_accusative(canonical) + suffix
    return piece_text


def _capture_object_case_patterns(piece_word: str) -> tuple[str, str]:
    """
    Возвращает:
      1) допустимую форму названия фигуры как прямого объекта взятия;
      2) ошибочную именительную форму.

    Цвет может стоять перед фигурой:
      "берёт чёрного ферзя" — допустимо,
      "берёт чёрный ферзь" — ошибка.
    """
    accusative_forms = {
        "пешка": r"пешку",
        "конь": r"коня",
        "слон": r"слона",
        "ладья": r"ладью",
        "ферзь": r"ферзя",
        "король": r"короля",
    }
    nominative_forms = {
        "пешка": r"пешка",
        "конь": r"конь",
        "слон": r"слон",
        "ладья": r"ладья",
        "ферзь": r"ферзь",
        "король": r"король",
    }
    return (
        accusative_forms.get(piece_word, re.escape(piece_word)),
        nominative_forms.get(piece_word, re.escape(piece_word)),
    )
