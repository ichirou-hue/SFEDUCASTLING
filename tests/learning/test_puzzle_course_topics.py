from backend.api_gateway.routes.learning import (
    COURSE_TOPIC_PUZZLE_THEMES,
    _puzzle_matches_course_topic,
)
from backend.services.course_topics import primary_course_topic_for_puzzle


def test_course_topics_cover_all_training_modules():
    assert set(COURSE_TOPIC_PUZZLE_THEMES) == {
        "pieces",
        "king-safety",
        "endgames",
        "tactics",
        "openings",
        "middlegame",
    }


def test_puzzle_matches_related_course_topic():
    assert _puzzle_matches_course_topic(
        {"themes": ["crushing", "advancedPawn", "endgame"]},
        "pieces",
    )
    assert _puzzle_matches_course_topic(
        {"themes": ["mate", "mateIn2", "middlegame"]},
        "king-safety",
    )
    assert _puzzle_matches_course_topic(
        {"themes": ["promotion", "endgame"]},
        "endgames",
    )


def test_puzzle_does_not_match_unrelated_topic():
    puzzle = {"themes": ["bishopEndgame", "endgame"]}
    assert _puzzle_matches_course_topic(puzzle, "pieces")
    assert not _puzzle_matches_course_topic(puzzle, "tactics")
    assert not _puzzle_matches_course_topic(puzzle, "unknown")


def test_tactics_match_without_piece_check():
    # Фильтр «профильная фигура ходит в решении» упразднён вместе
    # со слиянием модулей: тактика относится к tactics, кем бы ни ходили.
    no_knight = {"themes": ["fork"], "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"}
    assert _puzzle_matches_course_topic(no_knight, "tactics")

    with_bishop = {"themes": ["pin"], "fen": "4k3/8/8/8/8/8/8/3BK3 w - - 0 1"}
    assert _puzzle_matches_course_topic(with_bishop, "tactics")


def test_primary_topic_needs_no_piece():
    knight_moves = {
        "themes": ["fork"],
        "fen": "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
        "moves": "b1c3",
    }
    king_moves = {
        "themes": ["fork"],
        "fen": "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
        "moves": "e1e2",
    }
    assert primary_course_topic_for_puzzle(knight_moves) == "tactics"
    assert primary_course_topic_for_puzzle(king_moves) == "tactics"


def test_openings_topic_matches_opening_puzzles():
    assert _puzzle_matches_course_topic({"themes": ["opening", "trap"]}, "openings")
    assert primary_course_topic_for_puzzle({"themes": ["opening"]}) == "openings"
    assert not _puzzle_matches_course_topic({"themes": ["fork"]}, "openings")
