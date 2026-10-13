"""Общее соответствие модулей курса и Lichess themes."""

from __future__ import annotations

from typing import Any

import chess


COURSE_TOPIC_META: dict[str, dict[str, Any]] = {
    "pawn": {
        "title": "Пешка",
        "themes": {"advancedPawn", "pawnEndgame"},
    },
    "knight": {
        "title": "Конь",
        "themes": {"knightEndgame", "fork"},
    },
    "bishop": {
        "title": "Слон",
        "themes": {"bishopEndgame", "pin", "skewer"},
    },
    "rook": {
        "title": "Ладья",
        "themes": {"rookEndgame", "backRankMate"},
    },
    "queen": {
        "title": "Ферзь",
        "themes": {"queenEndgame", "attraction", "sacrifice"},
    },
    "king": {
        "title": "Король",
        "themes": {"exposedKing", "kingsideAttack", "defensiveMove"},
    },
    "special-rules": {
        "title": "Специальные правила",
        "themes": {"enPassant", "promotion", "underPromotion"},
    },
    "check-mate-stalemate": {
        "title": "Шах, мат и пат",
        "themes": {
            "mate",
            "mateIn1",
            "mateIn2",
            "mateIn3",
            "mateIn4",
            "mateIn5",
            "smotheredMate",
            "backRankMate",
            "doubleCheck",
        },
    },
    "endgames": {
        "title": "Эндшпиль",
        "themes": {"endgame", "opposition", "passedPawn"},
    },
}

COURSE_TOPIC_PUZZLE_THEMES: dict[str, set[str]] = {
    slug: set(meta["themes"]) for slug, meta in COURSE_TOPIC_META.items()
}

# Модули курса названы по фигурам, поэтому пазл относится к такому модулю,
# только если эта фигура реально участвует в решении: делает хотя бы один ход.
# Одного присутствия на доске мало — тактические темы (pin, skewer, fork,
# attraction, sacrifice) выполняет любая фигура, а не обязательно «своя».
PIECE_BY_MODULE: dict[str, str] = {
    "pawn": "P",
    "knight": "N",
    "bishop": "B",
    "rook": "R",
    "queen": "Q",
    "king": "K",
}

_PIECE_TYPE_BY_SYMBOL = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}


def _fen_has_piece(fen: str | None, symbol: str) -> bool:
    """Есть ли на доске фигура указанного типа (цвет не важен) по FEN.

    Без FEN считаем условие выполненным, чтобы не отсекать записи, для
    которых проверить позицию нельзя.
    """
    if not fen:
        return True
    placement = fen.split(" ", 1)[0].lower()
    return symbol.lower() in placement


def _pieces_used_in_solution(fen: str | None, moves: Any) -> set[int] | None:
    """Типы фигур, которые делают ход в решении (по UCI-последовательности).

    Возвращает None, если позицию или ходы разобрать нельзя: тогда вызывающий
    код откатывается к простой проверке наличия фигуры по FEN.
    """
    if not fen or not moves:
        return None
    sequence = "".join(str(moves).split())
    if len(sequence) < 4:
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    used: set[int] = set()
    parsed_any = False
    for i in range(0, len(sequence) - 3, 4):
        try:
            move = chess.Move.from_uci(sequence[i : i + 4])
        except ValueError:
            break
        piece = board.piece_at(move.from_square)
        if piece is None:
            break
        try:
            board.push(move)
        except Exception:
            break
        used.add(piece.piece_type)
        parsed_any = True
    return used if parsed_any else None


def _module_piece_ok(puzzle: dict, topic: str) -> bool:
    """Участвует ли «профильная» фигура модуля в решении пазла."""
    symbol = PIECE_BY_MODULE.get(topic)
    if symbol is None:
        return True
    used = _pieces_used_in_solution(puzzle.get("fen"), puzzle.get("moves"))
    if used is not None:
        return _PIECE_TYPE_BY_SYMBOL[symbol] in used
    return _fen_has_piece(puzzle.get("fen"), symbol)

# Один пазл может затрагивать несколько модулей. Для статистики B1 одна
# puzzle_attempt относится к одной основной теме, чтобы не удваивать попытки.
THEME_TO_PRIMARY_TOPIC: list[tuple[str, str]] = [
    ("enPassant", "special-rules"),
    ("promotion", "special-rules"),
    ("underPromotion", "special-rules"),
    ("pawnEndgame", "pawn"),
    ("advancedPawn", "pawn"),
    ("knightEndgame", "knight"),
    ("bishopEndgame", "bishop"),
    ("rookEndgame", "rook"),
    ("queenEndgame", "queen"),
    ("mateIn1", "check-mate-stalemate"),
    ("mateIn2", "check-mate-stalemate"),
    ("mateIn3", "check-mate-stalemate"),
    ("mateIn4", "check-mate-stalemate"),
    ("mateIn5", "check-mate-stalemate"),
    ("mate", "check-mate-stalemate"),
    ("smotheredMate", "check-mate-stalemate"),
    ("backRankMate", "check-mate-stalemate"),
    ("doubleCheck", "check-mate-stalemate"),
    ("fork", "knight"),
    ("pin", "bishop"),
    ("skewer", "bishop"),
    ("attraction", "queen"),
    ("sacrifice", "queen"),
    ("exposedKing", "king"),
    ("kingsideAttack", "king"),
    ("defensiveMove", "king"),
    ("endgame", "endgames"),
    ("opposition", "endgames"),
    ("passedPawn", "endgames"),
]


def puzzle_matches_course_topic(puzzle: dict, topic: str) -> bool:
    wanted = COURSE_TOPIC_PUZZLE_THEMES.get(topic)
    if not wanted:
        return False
    if not set(puzzle.get("themes") or []) & wanted:
        return False
    return _module_piece_ok(puzzle, topic)


def primary_course_topic_for_puzzle(puzzle: dict | None) -> str | None:
    if not puzzle:
        return None
    themes = set(puzzle.get("themes") or [])
    for theme, slug in THEME_TO_PRIMARY_TOPIC:
        if theme not in themes:
            continue
        if not _module_piece_ok(puzzle, slug):
            # Тема совпала, но профильная фигура модуля не участвует в
            # решении — пробуем следующую подходящую тему.
            continue
        return slug
    return None
