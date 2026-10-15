"""Чистая логика B2: пороги difficulty и рейтинговые диапазоны пазлов."""

from __future__ import annotations

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 3
DEFAULT_DIFFICULTY = 2

PUZZLE_RATING_BANDS: dict[int, tuple[int, int]] = {
    1: (0, 1100),
    2: (1101, 1700),
    3: (1701, 9999),
}


def next_difficulty(current: int, accuracy: float | None) -> int:
    current = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(current)))
    if accuracy is None:
        return current
    if float(accuracy) > 80.0:
        return min(MAX_DIFFICULTY, current + 1)
    if float(accuracy) < 50.0:
        return max(MIN_DIFFICULTY, current - 1)
    return current


def rating_band_for_difficulty(difficulty: int) -> tuple[int, int]:
    difficulty = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(difficulty)))
    return PUZZLE_RATING_BANDS[difficulty]


def difficulty_for_rating(rating: int | float | None) -> int:
    """Стартовая сложность по рейтингу — обратная к rating_band_for_difficulty.

    Используется, чтобы задать стартовую B2-сложность тем из рейтинга,
    полученного после анкеты/входного теста (вместо нейтральной 2).
    Без рейтинга возвращает нейтральный уровень по умолчанию.
    """
    if rating is None:
        return DEFAULT_DIFFICULTY
    try:
        value = int(rating)
    except (TypeError, ValueError):
        return DEFAULT_DIFFICULTY

    for difficulty in range(MIN_DIFFICULTY, MAX_DIFFICULTY + 1):
        min_rating, max_rating = PUZZLE_RATING_BANDS[difficulty]
        if min_rating <= value <= max_rating:
            return difficulty
    return MIN_DIFFICULTY if value < PUZZLE_RATING_BANDS[MIN_DIFFICULTY][0] else MAX_DIFFICULTY


def puzzle_matches_difficulty(puzzle: dict, difficulty: int) -> bool:
    min_rating, max_rating = rating_band_for_difficulty(difficulty)
    try:
        rating = int(puzzle.get("rating") or 0)
    except (TypeError, ValueError):
        return False
    return min_rating <= rating <= max_rating
