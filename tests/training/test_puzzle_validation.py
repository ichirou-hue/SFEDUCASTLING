"""Проверки валидатора контента и инвариантов курса и базы пазлов."""

import json
from pathlib import Path

import chess
import pytest

from backend.services.course_topics import (
    COURSE_TOPIC_META,
    _theme_piece_ok,
    puzzle_matches_course_topic,
)
from backend.services.puzzle_validation import (
    PIECE_THEMES,
    pieces_used_in_solution,
    puzzle_themes_piece_ok,
    validate_puzzle,
    validate_training_task,
)
from backend.services.training_course_data import LESSONS_BY_MODULE

PUZZLES_PATH = Path(__file__).resolve().parents[2] / "backend" / "knowledge" / "puzzles.json"


def _all_puzzles():
    with open(PUZZLES_PATH, encoding="utf-8") as fh:
        return json.load(fh)["puzzles"]


def test_pieces_used_in_solution_handles_promotion():
    used = pieces_used_in_solution(
        "8/P7/8/8/8/8/7k/7K w - - 0 1", "a7a8q"
    )
    assert used == {chess.PAWN}


def test_piece_theme_mismatch_is_detected():
    rook_theme_king_moves = {
        "id": "mismatch",
        "themes": ["rookEndgame", "endgame"],
        "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "moves": "e1e2",
    }
    assert not puzzle_themes_piece_ok(rook_theme_king_moves)
    assert validate_puzzle(rook_theme_king_moves, check_piece=True)
    assert validate_puzzle(rook_theme_king_moves, check_piece=False) == []


def test_piece_theme_match_is_accepted():
    rook_moves = {
        "themes": ["rookEndgame", "endgame"],
        "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "moves": "a1a2",
    }
    assert puzzle_themes_piece_ok(rook_moves)


def test_non_piece_themes_are_not_restricted():
    assert puzzle_themes_piece_ok(
        {"themes": ["fork"], "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "moves": "a1a2"}
    )


def test_validate_puzzle_rejects_illegal_sequence():
    bad = {
        "id": "x",
        "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "moves": "a1a7 e8e7",  # второй ход нелегален
        "themes": [],
    }
    assert validate_puzzle(bad, check_piece=False)


def test_validate_puzzle_accepts_mate_in_one():
    good = {
        "id": "m1",
        "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
        "moves": "e1e8",
        "themes": ["mateIn1"],
    }
    assert validate_puzzle(good) == []


def test_validate_training_task_rejects_wrong_target_piece():
    task = {
        "task_type": "make_move",
        "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "source_square": "e1",
        "payload": {"piece": "N", "mode": "all_legal_moves"},
    }
    assert validate_training_task(task)


def test_all_course_tasks_are_chess_correct():
    for module_slug, lessons in LESSONS_BY_MODULE.items():
        for lesson in lessons:
            for task in lesson["tasks"]:
                errors = validate_training_task(task)
                assert not errors, (
                    f"{module_slug}/{lesson['slug']}/{task['title']}: {errors}"
                )


def test_course_move_tasks_declare_target_piece_on_source():
    for lessons in LESSONS_BY_MODULE.values():
        for lesson in lessons:
            for task in lesson["tasks"]:
                if task["task_type"] not in {"select_squares", "make_move"}:
                    continue
                board = chess.Board(task["fen"])
                source = chess.parse_square(task["source_square"])
                piece = board.piece_at(source)
                assert piece is not None
                expected = task["payload"].get("piece")
                if expected:
                    assert piece.symbol().upper() == expected, task["title"]


def test_all_puzzles_are_legal():
    for puzzle in _all_puzzles():
        errors = validate_puzzle(puzzle, check_piece=False)
        assert not errors, f"{puzzle.get('id')}: {errors}"


def test_served_puzzles_respect_target_piece():
    for puzzle in _all_puzzles():
        for topic, meta in COURSE_TOPIC_META.items():
            if not puzzle_matches_course_topic(puzzle, topic):
                continue
            matched = set(puzzle.get("themes") or []) & set(meta["themes"])
            piece_themes = {theme for theme in matched if theme in PIECE_THEMES}
            if not piece_themes:
                continue
            assert _theme_piece_ok(puzzle, piece_themes), (
                f"{puzzle['id']} подаётся в {topic}, но фигура темы не ходит"
            )


def test_mismatched_puzzle_is_not_served_by_piece_topic():
    # '0POVV' помечен pawnEndgame, но пешка в решении не ходит.
    puzzle = next((p for p in _all_puzzles() if p["id"] == "0POVV"), None)
    if puzzle is None:
        pytest.skip("контрольный пазл отсутствует в базе")
    assert not puzzle_themes_piece_ok(puzzle)
    assert not puzzle_matches_course_topic(puzzle, "pieces")
