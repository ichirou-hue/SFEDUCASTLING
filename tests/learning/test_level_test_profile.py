"""Быстрые unit-тесты расчёта рейтинга/группы для level-test.

level_test_result — legacy preview без записи в профиль: Elo от начальных
1000 (k=72) по rating каждого пазла, затем группа из RATING_GROUPS.
DB-идемпотентность /submit дополнительно проверяется интеграционно после
alembic upgrade head: повторный submit одного test_id должен вернуть
already_submitted=true и тот же band.
"""

from backend.api_gateway import state
from backend.api_gateway.routes.learning import level_test_result


def _puzzle(pid: str, rating: int) -> dict:
    return {"id": pid, "rating": rating}


PUZZLES = [
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


def _answers(correct: bool) -> list[dict]:
    return [{"puzzle_id": p["id"], "correct": correct} for p in PUZZLES]


def test_all_correct_reaches_amateur_level(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})

    result = level_test_result(_answers(correct=True))

    assert result["rating"] == 1307
    assert result["level"] == 2
    assert result["band"] == 2
    assert result["result"]["name"] == "Любительский"
    assert result["result"]["rating"] == 1307


def test_all_wrong_stays_beginner_level(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})

    result = level_test_result(_answers(correct=False))

    assert result["rating"] == 774
    assert result["level"] == 1
    assert result["band"] == 1
    assert result["result"]["name"] == "Начальный"


def test_group_consistent_with_rating(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})

    for correct in (True, False):
        result = level_test_result(_answers(correct=correct))
        assert 0 <= result["rating"] <= 3500
        assert result["result"]["rating"] == result["rating"]
        assert result["level"] == result["band"] == result["result"]["level"]
