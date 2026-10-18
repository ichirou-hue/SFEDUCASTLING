from types import SimpleNamespace

from backend.services.adaptive_logic import select_adaptive_puzzles
from backend.services.difficulty_logic import (
    initial_difficulty_for_rating,
    next_difficulty,
    puzzle_matches_difficulty,
    rating_band_for_difficulty,
)
from backend.services.dynamic_difficulty import effective_player_rating


def test_thresholds_are_strict():
    assert next_difficulty(2, 81.0) == 3
    assert next_difficulty(2, 80.0) == 2
    assert next_difficulty(2, 50.0) == 2
    assert next_difficulty(2, 49.9) == 1


def test_difficulty_is_bounded():
    assert next_difficulty(3, 100.0) == 3
    assert next_difficulty(1, 0.0) == 1
    assert next_difficulty(2, None) == 2


def test_rating_bands():
    assert rating_band_for_difficulty(1) == (0, 1100)
    assert rating_band_for_difficulty(2) == (1101, 1700)
    assert rating_band_for_difficulty(3) == (1701, 9999)
    assert puzzle_matches_difficulty({"rating": 900}, 1)
    assert puzzle_matches_difficulty({"rating": 1500}, 2)
    assert puzzle_matches_difficulty({"rating": 2100}, 3)


def test_starting_difficulty_follows_player_rating():
    assert initial_difficulty_for_rating(None) == 1
    assert initial_difficulty_for_rating(1100) == 1
    assert initial_difficulty_for_rating(1101) == 2
    assert initial_difficulty_for_rating(1700) == 2
    assert initial_difficulty_for_rating(1701) == 3
    assert not puzzle_matches_difficulty({"rating": None}, 1)


def test_confirmed_rating_takes_precedence_over_onboarding_estimate():
    user = SimpleNamespace(assessment_completed_at=None, elo=None, rating_estimate=1800)
    assert effective_player_rating(user) == 1800
    user.elo = 850
    user.assessment_completed_at = "completed"
    assert effective_player_rating(user) == 850


def test_adaptive_selection_prefers_current_topic_difficulty():
    puzzles = [
        {
            "id": "easy-knight",
            "themes": ["fork"],
            "rating": 800,
        },
        {
            "id": "medium-knight",
            "themes": ["fork"],
            "rating": 1500,
        },
    ]
    picked, allocation = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "knight"}],
        strong_topics=[],
        count=1,
        topic_difficulties={"knight": 1},
    )
    assert picked[0]["id"] == "easy-knight"
    assert picked[0]["adaptive_difficulty"] == 1
    assert picked[0]["difficulty_match"] is True
    assert allocation["difficulty_matched"] == 1
    assert allocation["difficulty_fallback"] == 0


def test_same_topic_fallback_preserves_topic_when_rating_band_is_empty():
    puzzles = [
        {
            "id": "hard-knight",
            "themes": ["fork"],
            "rating": 2100,
        }
    ]
    picked, allocation = select_adaptive_puzzles(
        puzzles,
        weak_topics=[{"slug": "knight"}],
        strong_topics=[],
        count=1,
        topic_difficulties={"knight": 1},
    )
    assert picked[0]["adaptive_topic"] == "knight"
    assert picked[0]["adaptive_difficulty"] == 1
    assert picked[0]["difficulty_match"] is False
    assert allocation["difficulty_matched"] == 0
    assert allocation["difficulty_fallback"] == 1


def test_updated_at_column_is_timezone_aware():
    from backend.models.user_theme_difficulty import UserThemeDifficulty

    assert UserThemeDifficulty.__table__.c.updated_at.type.timezone is True
