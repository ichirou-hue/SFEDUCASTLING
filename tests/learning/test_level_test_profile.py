"""Быстрые unit-тесты расчёта band для level-test.

DB-идемпотентность /submit дополнительно проверяется интеграционно после
alembic upgrade head: повторный submit одного test_id должен вернуть
already_submitted=true и тот же band.
"""

from backend.api_gateway import state
from backend.api_gateway.routes.learning import _score_level_test


def _puzzle(pid: str, rating: int) -> dict:
    return {"id": pid, "rating": rating}


def test_score_level_test_returns_1500_band(monkeypatch):
    puzzles = [
        _puzzle("l1a", 600),
        _puzzle("l1b", 650),
        _puzzle("l1c", 700),
        _puzzle("l2a", 1000),
        _puzzle("l2b", 1050),
        _puzzle("l2c", 1100),
        _puzzle("l3a", 1400),
        _puzzle("l3b", 1450),
        _puzzle("l3c", 1500),
    ]
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": puzzles})

    answers = [
        {"puzzle_id": puzzle["id"], "correct": True}
        for puzzle in puzzles
    ]

    result = _score_level_test(answers)

    assert result["level"] == 3
    assert result["band"] == 1500
    assert result["result"]["name"] == "Клубный"


def test_score_level_test_default_band_is_500(monkeypatch):
    puzzles = [
        _puzzle("a", 600),
        _puzzle("b", 650),
        _puzzle("c", 700),
    ]
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": puzzles})

    answers = [
        {"puzzle_id": puzzle["id"], "correct": False}
        for puzzle in puzzles
    ]

    result = _score_level_test(answers)

    assert result["level"] == 1
    assert result["band"] == 500
