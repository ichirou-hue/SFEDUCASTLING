from backend.api_gateway.routes.learning import (
    COURSE_TOPIC_PUZZLE_THEMES,
    _puzzle_matches_course_topic,
)


def test_course_topics_cover_all_training_modules():
    assert set(COURSE_TOPIC_PUZZLE_THEMES) == {
        "pawn",
        "knight",
        "bishop",
        "rook",
        "queen",
        "king",
        "special-rules",
        "check-mate-stalemate",
    }


def test_puzzle_matches_related_course_topic():
    assert _puzzle_matches_course_topic(
        {"themes": ["crushing", "advancedPawn", "endgame"]},
        "pawn",
    )
    assert _puzzle_matches_course_topic(
        {"themes": ["mate", "mateIn2", "middlegame"]},
        "check-mate-stalemate",
    )
    assert _puzzle_matches_course_topic(
        {"themes": ["promotion", "endgame"]},
        "special-rules",
    )


def test_puzzle_does_not_match_unrelated_topic():
    puzzle = {"themes": ["bishopEndgame", "endgame"]}
    assert _puzzle_matches_course_topic(puzzle, "bishop")
    assert not _puzzle_matches_course_topic(puzzle, "knight")
    assert not _puzzle_matches_course_topic(puzzle, "unknown")
