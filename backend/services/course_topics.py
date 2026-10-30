"""Общее соответствие модулей курса и Lichess themes."""

from __future__ import annotations

from typing import Any

import chess

from backend.services.puzzle_validation import (
    PIECE_THEMES,
    pieces_used_in_solution,
)

COURSE_TOPIC_META: dict[str, dict[str, Any]] = {
    "pieces": {
        "title": "Фигуры: ходы и взятия",
        "themes": {
            "advancedPawn",
            "pawnEndgame",
            "knightEndgame",
            "bishopEndgame",
            "rookEndgame",
            "queenEndgame",
        },
    },
    "king-safety": {
        "title": "Шах, мат и безопасность короля",
        "themes": {
            "mate",
            "mateIn1",
            "mateIn2",
            "mateIn3",
            "mateIn4",
            "mateIn5",
            "smotheredMate",
            "backRankMate",
            "exposedKing",
            "kingsideAttack",
            "defensiveMove",
        },
    },
    "endgames": {
        "title": "Эндшпиль",
        "themes": {
            "endgame",
            "opposition",
            "passedPawn",
            "enPassant",
            "promotion",
            "underPromotion",
        },
    },
    "tactics": {
        "title": "Тактические приёмы",
        "themes": {
            "fork",
            "pin",
            "skewer",
            "attraction",
            "sacrifice",
            "doubleCheck",
            "hangingPiece",
            "capturingDefender",
            "discoveredAttack",
            "discoveredCheck",
            "deflection",
            "intermezzo",
            "clearance",
            "trappedPiece",
        },
    },
    "openings": {
        "title": "Дебюты",
        "themes": {"opening"},
    },
    "middlegame": {
        "title": "Миттельшпиль: стратегия",
        "themes": {
            "middlegame",
            "quietMove",
            "kingsideAttack",
            "queensideAttack",
            "exposedKing",
        },
    },
}

COURSE_TOPIC_PUZZLE_THEMES: dict[str, set[str]] = {
    slug: set(meta["themes"]) for slug, meta in COURSE_TOPIC_META.items()
}

# Раньше модули назывались по фигурам, и пазл относился к такому модулю,
# только если эта фигура реально ходила в решении. После слияния модулей
# (10 -> 5) профильных фигур у тем больше нет: тактики выделены в tactics,
# пешечные окончания — в pieces/endgames. Словарь оставлен пустым намеренно:
# _module_piece_ok возвращает True, когда символа нет.
PIECE_BY_MODULE: dict[str, str] = {}

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


def _theme_piece_ok(puzzle: dict, themes: set[str]) -> bool:
    """Участвуют ли названные темами фигуры в решении пазла.

    Темы вида pawnEndgame/knightEndgame/rookEndgame/promotion называют
    конкретную фигуру. Если такая тема есть, а фигура в решении не ходит,
    пазл помечен некорректно и в подбор по этой теме не попадает.
    """
    wanted = {PIECE_THEMES[theme] for theme in themes if theme in PIECE_THEMES}
    if not wanted:
        return True
    used = pieces_used_in_solution(puzzle.get("fen"), puzzle.get("moves"))
    if used is not None:
        return all(_PIECE_TYPE_BY_SYMBOL[symbol] in used for symbol in wanted)
    return all(_fen_has_piece(puzzle.get("fen"), symbol) for symbol in wanted)

# Один пазл может затрагивать несколько модулей. Для статистики B1 одна
# puzzle_attempt относится к одной основной теме, чтобы не удваивать попытки.
# Порядок важен: побеждает первая совпавшая тема.
THEME_TO_PRIMARY_TOPIC: list[tuple[str, str]] = [
    ("enPassant", "endgames"),
    ("promotion", "endgames"),
    ("underPromotion", "endgames"),
    ("pawnEndgame", "pieces"),
    ("advancedPawn", "pieces"),
    ("knightEndgame", "pieces"),
    ("bishopEndgame", "pieces"),
    ("rookEndgame", "pieces"),
    ("queenEndgame", "pieces"),
    ("mateIn1", "king-safety"),
    ("mateIn2", "king-safety"),
    ("mateIn3", "king-safety"),
    ("mateIn4", "king-safety"),
    ("mateIn5", "king-safety"),
    ("mate", "king-safety"),
    ("smotheredMate", "king-safety"),
    ("backRankMate", "king-safety"),
    ("fork", "tactics"),
    ("pin", "tactics"),
    ("skewer", "tactics"),
    ("attraction", "tactics"),
    ("sacrifice", "tactics"),
    ("doubleCheck", "tactics"),
    ("hangingPiece", "tactics"),
    ("capturingDefender", "tactics"),
    ("discoveredAttack", "tactics"),
    ("discoveredCheck", "tactics"),
    ("deflection", "tactics"),
    ("intermezzo", "tactics"),
    ("clearance", "tactics"),
    ("trappedPiece", "tactics"),
    ("opening", "openings"),
    ("exposedKing", "king-safety"),
    ("kingsideAttack", "king-safety"),
    ("defensiveMove", "king-safety"),
    ("endgame", "endgames"),
    ("opposition", "endgames"),
    ("passedPawn", "endgames"),
    ("middlegame", "middlegame"),
    ("quietMove", "middlegame"),
    ("queensideAttack", "middlegame"),
]


def puzzle_matches_course_topic(puzzle: dict, topic: str) -> bool:
    wanted = COURSE_TOPIC_PUZZLE_THEMES.get(topic)
    if not wanted:
        return False
    matched = set(puzzle.get("themes") or []) & wanted
    if not matched:
        return False
    if not _theme_piece_ok(puzzle, matched):
        return False
    return _module_piece_ok(puzzle, topic)


def primary_course_topic_for_puzzle(puzzle: dict | None) -> str | None:
    if not puzzle:
        return None
    themes = set(puzzle.get("themes") or [])
    for theme, slug in THEME_TO_PRIMARY_TOPIC:
        if theme not in themes:
            continue
        if not _theme_piece_ok(puzzle, {theme}):
            # Тема называет фигуру, которой в решении нет — пропускаем её.
            continue
        if not _module_piece_ok(puzzle, slug):
            # Тема совпала, но профильная фигура модуля не участвует в
            # решении — пробуем следующую подходящую тему.
            continue
        return slug
    return None
