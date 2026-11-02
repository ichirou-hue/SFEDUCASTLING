"""Текущая серия решённых пазлов.

Правило: серия растёт за каждую верную попытку и обнуляется любой ошибкой.
Отдельно от «серии дней», где день засчитывается за 3+ разных пазла
или задание курса (см. tests/training/test_training_progress_contract.py).
"""

import asyncio
from datetime import UTC, datetime, timedelta

from backend.services.adaptive_training import compute_learning_progress
from backend.services.user_records import build_user_records


class _FakeDB:
    """Заглушка БД: db.scalar(...) отдаёт значения по порядку вызовов."""

    def __init__(self, scalars):
        self._scalars = list(scalars)

    async def scalar(self, _statement):
        return self._scalars.pop(0)


class _Attempt:
    def __init__(self, id, correct, created_at, puzzle_id="p1"):
        self.id = id
        self.correct = correct
        self.created_at = created_at
        self.puzzle_id = puzzle_id


def _run(coro):
    return asyncio.run(coro)


def test_puzzle_streak_counts_correct_attempts_after_last_error():
    # db.scalar(...) по порядку: всего, верных, взятых, решённых,
    # id последней ошибки, верных после неё.
    db = _FakeDB([10, 7, 5, 4, 6, 3])

    result = _run(compute_learning_progress(db, user_id=1))

    assert result["puzzle_streak"] == 3
    # Остальные счётчики не изменились.
    assert result["attempts"] == 10
    assert result["correct_attempts"] == 7
    assert result["accuracy"] == 80.0


def test_puzzle_streak_is_zero_for_new_user():
    db = _FakeDB([0, 0, 0, 0, None, 0])

    result = _run(compute_learning_progress(db, user_id=1))

    assert result["puzzle_streak"] == 0
    assert result["accuracy"] is None


def test_records_reset_current_puzzle_streak_on_error():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    attempts = [
        _Attempt(1, True, now - timedelta(minutes=5), "a"),
        _Attempt(2, True, now - timedelta(minutes=4), "b"),
        _Attempt(3, False, now - timedelta(minutes=3), "c"),
        _Attempt(4, True, now - timedelta(minutes=2), "d"),
    ]

    result = build_user_records([], attempts, now=now)

    assert result["records"]["best_puzzle_streak"] == 2
    assert result["records"]["current_puzzle_streak"] == 1


def test_records_current_puzzle_streak_grows_without_errors():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    attempts = [
        _Attempt(1, True, now - timedelta(minutes=2), "a"),
        _Attempt(2, True, now - timedelta(minutes=1), "b"),
    ]

    result = build_user_records([], attempts, now=now)

    assert result["records"]["best_puzzle_streak"] == 2
    assert result["records"]["current_puzzle_streak"] == 2
