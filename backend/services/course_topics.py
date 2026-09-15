"""Общее соответствие модулей курса и Lichess themes."""

from __future__ import annotations

from typing import Any


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
}

COURSE_TOPIC_PUZZLE_THEMES: dict[str, set[str]] = {
    slug: set(meta["themes"]) for slug, meta in COURSE_TOPIC_META.items()
}

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
]


def puzzle_matches_course_topic(puzzle: dict, topic: str) -> bool:
    wanted = COURSE_TOPIC_PUZZLE_THEMES.get(topic)
    if not wanted:
        return False
    return bool(set(puzzle.get("themes") or []) & wanted)


def primary_course_topic_for_puzzle(puzzle: dict | None) -> str | None:
    if not puzzle:
        return None
    themes = set(puzzle.get("themes") or [])
    for theme, slug in THEME_TO_PRIMARY_TOPIC:
        if theme in themes:
            return slug
    return None
