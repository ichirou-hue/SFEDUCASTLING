from backend.api_gateway.routes.learning import (
    COURSE_TOPIC_PUZZLE_THEMES,
    _puzzle_matches_course_topic,
)
from backend.services.course_topics import primary_course_topic_for_puzzle


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
        "endgames",
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


def test_piece_topic_presence_fallback_without_moves():
    # Без ходов решения проверка откатывается к наличию фигуры на доске.
    no_bishop = {"themes": ["skewer"], "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"}
    assert not _puzzle_matches_course_topic(no_bishop, "bishop")

    with_bishop = {"themes": ["pin"], "fen": "4k3/8/8/8/8/8/8/3BK3 w - - 0 1"}
    assert _puzzle_matches_course_topic(with_bishop, "bishop")


def test_primary_topic_skips_theme_without_piece():
    no_bishop = {"themes": ["skewer"], "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"}
    assert primary_course_topic_for_puzzle(no_bishop) is None

    with_bishop = {"themes": ["skewer"], "fen": "4k3/8/8/8/8/8/8/3BK3 w - - 0 1"}
    assert primary_course_topic_for_puzzle(with_bishop) == "bishop"


_KNIGHT_FEN = "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1"


def test_piece_topic_requires_piece_used_in_solution():
    # Конь стоит на доске (b1), но фигурная тема требует, чтобы он ходил.
    knight_moves = {"themes": ["fork"], "fen": _KNIGHT_FEN, "moves": "b1c3"}
    king_moves = {"themes": ["fork"], "fen": _KNIGHT_FEN, "moves": "e1e2"}
    assert _puzzle_matches_course_topic(knight_moves, "knight")
    assert not _puzzle_matches_course_topic(king_moves, "knight")


def test_primary_topic_requires_piece_used_in_solution():
    knight_moves = {"themes": ["fork"], "fen": _KNIGHT_FEN, "moves": "b1c3"}
    king_moves = {"themes": ["fork"], "fen": _KNIGHT_FEN, "moves": "e1e2"}
    assert primary_course_topic_for_puzzle(knight_moves) == "knight"
    assert primary_course_topic_for_puzzle(king_moves) is None
