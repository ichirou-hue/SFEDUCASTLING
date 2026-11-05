"""Быстрые unit-тесты расчёта рейтинга/группы для level-test.

level_test_result - legacy preview без записи в профиль: детерминированная
кусочно-линейная шкала от начальных 1000, затем группа из RATING_GROUPS.
Итог зависит только от числа верных ответов, а не от того, какие задачи
попали в набор.

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

# 9 задач: половина верных = 4.5, поэтому «всё верно» даёт полный прирост
# (+680 от стартовых 1000), «всё неверно» - полный промах (-300).
ALL_CORRECT_RATING = 1680
ALL_WRONG_RATING = 700


def _answers(correct: bool) -> list[dict]:
    return [{"puzzle_id": p["id"], "correct": correct} for p in PUZZLES]


def test_all_correct_reaches_club_level(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})

    result = level_test_result(_answers(correct=True))

    assert result["rating"] == ALL_CORRECT_RATING
    assert result["level"] == 3
    assert result["band"] == 3
    assert result["result"]["name"] == "Клубный"
    assert result["result"]["rating"] == ALL_CORRECT_RATING


def test_all_wrong_stays_beginner_level(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})

    result = level_test_result(_answers(correct=False))

    assert result["rating"] == ALL_WRONG_RATING
    assert result["level"] == 1
    assert result["band"] == 1
    assert result["result"]["name"] == "Начальный"


def test_result_does_not_depend_on_which_puzzles_were_offered(monkeypatch):
    """Главное требование: 0/20 всегда даёт одно и то же число.

    Раньше рейтинг считался Elo-ом по рейтингам конкретных выпавших задач,
    поэтому при 0 верных он гулял в диапазоне ~425–580.
    """
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})
    ratings = set()
    for variant in range(50):
        # Разные наборы задач с одинаковым числом верных ответов.
        offset = variant % len(PUZZLES)
        rotated = PUZZLES[offset:] + PUZZLES[:offset]
        answers = [{"puzzle_id": p["id"], "correct": False} for p in rotated]
        ratings.add(level_test_result(answers)["rating"])

    assert ratings == {ALL_WRONG_RATING}


def test_rating_grows_monotonically_with_correct_answers(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})
    ratings = []
    for correct_count in range(len(PUZZLES) + 1):
        answers = [
            {"puzzle_id": p["id"], "correct": index < correct_count}
            for index, p in enumerate(PUZZLES)
        ]
        ratings.append(level_test_result(answers)["rating"])

    assert ratings == sorted(ratings)
    assert len(set(ratings)) > 1
    assert ratings[0] == ALL_WRONG_RATING
    assert ratings[-1] == ALL_CORRECT_RATING


def test_group_consistent_with_rating(monkeypatch):
    monkeypatch.setattr(state, "puzzle_base", {"puzzles": PUZZLES})

    for correct in (True, False):
        result = level_test_result(_answers(correct=correct))
        assert 0 <= result["rating"] <= 3500
        assert result["result"]["rating"] == result["rating"]
        assert result["level"] == result["band"] == result["result"]["level"]
