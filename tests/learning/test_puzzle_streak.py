"""Текущая серия решённых пазлов.

Правило: серия растёт за ПЕРВОЕ решение пазла (его ещё ни разу не
пробовали) и обнуляется любой ошибкой или решением с подсказкой.
Повторное решение уже пробованного пазла нейтрально: серию не растягивает
и не обрывает (ошибка уже обнулила её раньше).
Отдельно от «серии дней», где день засчитывается за 3+ разных пазла
или задание курса (см. tests/training/test_training_progress_contract.py).
"""

import asyncio
from datetime import UTC, datetime, timedelta

from backend.services.adaptive_training import compute_learning_progress
from backend.services.user_records import build_user_records


class _FakeDB:
    """Заглушка БД: db.scalar(...) отдаёт значения по порядку вызовов,
    db.scalars(...) - строки попыток пазлов для пересчёта серии."""

    def __init__(self, scalars, attempts=()):
        self._scalars = list(scalars)
        self._attempts = list(attempts)

    async def scalar(self, _statement):
        return self._scalars.pop(0)

    async def scalars(self, _statement):
        return self._attempts


class _Attempt:
    def __init__(self, id, correct, created_at, puzzle_id="p1", hint_used=False):
        self.id = id
        self.correct = correct
        self.created_at = created_at
        self.puzzle_id = puzzle_id
        self.hint_used = hint_used


def _run(coro):
    return asyncio.run(coro)


def test_puzzle_streak_counts_first_solves_after_last_error():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    # db.scalar(...) по порядку: всего, верных, взятых, решённых.
    db = _FakeDB(
        [10, 7, 5, 4],
        attempts=[
            _Attempt(1, True, now - timedelta(minutes=6), "a"),
            _Attempt(2, True, now - timedelta(minutes=5), "b"),
            _Attempt(3, True, now - timedelta(minutes=4), "c"),
            _Attempt(4, False, now - timedelta(minutes=3), "d"),
            _Attempt(5, True, now - timedelta(minutes=2), "e"),
            _Attempt(6, True, now - timedelta(minutes=1), "f"),
        ],
    )

    result = _run(compute_learning_progress(db, user_id=1))

    assert result["puzzle_streak"] == 2
    # Остальные счётчики не изменились.
    assert result["attempts"] == 10
    assert result["correct_attempts"] == 7
    assert result["accuracy"] == 80.0


def test_puzzle_streak_is_zero_for_new_user():
    db = _FakeDB([0, 0, 0, 0])

    result = _run(compute_learning_progress(db, user_id=1))

    assert result["puzzle_streak"] == 0
    assert result["accuracy"] is None


def test_progress_retry_after_error_is_neutral():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    db = _FakeDB(
        [4, 3, 3, 2],
        attempts=[
            _Attempt(1, True, now - timedelta(minutes=4), "a"),
            _Attempt(2, False, now - timedelta(minutes=3), "b"),
            _Attempt(3, True, now - timedelta(minutes=2), "b"),
            _Attempt(4, True, now - timedelta(minutes=1), "c"),
        ],
    )

    result = _run(compute_learning_progress(db, user_id=1))

    # Ошибка на b обнулила серию, повторное решение b её не поднимает,
    # а новый пазл c поднимает до 1.
    assert result["puzzle_streak"] == 1


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


def test_records_hint_solve_resets_current_puzzle_streak():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    attempts = [
        _Attempt(1, True, now - timedelta(minutes=5), "a"),
        _Attempt(2, True, now - timedelta(minutes=4), "b"),
        # Решение с подсказкой засчитывается, но обрывает серию.
        _Attempt(3, True, now - timedelta(minutes=3), "c", hint_used=True),
        _Attempt(4, True, now - timedelta(minutes=2), "d"),
    ]

    result = build_user_records([], attempts, now=now)

    assert result["records"]["best_puzzle_streak"] == 2
    assert result["records"]["current_puzzle_streak"] == 1
    # В дневном графике решение с подсказкой остаётся верным: все 4 попытки
    # в один день (Москва) засчитаны как верные.
    day_item = next(
        item for item in result["activity"]["puzzles"] if item["date"] == "2026-09-14"
    )
    assert day_item["correct"] == 4
    assert day_item["incorrect"] == 0


def test_records_retry_of_failed_puzzle_does_not_grow_streak():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    attempts = [
        _Attempt(1, True, now - timedelta(minutes=6), "a"),
        _Attempt(2, False, now - timedelta(minutes=5), "b"),
        _Attempt(3, True, now - timedelta(minutes=4), "b"),
        _Attempt(4, True, now - timedelta(minutes=3), "c"),
    ]

    result = build_user_records([], attempts, now=now)

    # Решение b после ошибки на нём серию не растягивает: она остаётся 0,
    # а следующий новый пазл c даёт 1.
    assert result["records"]["best_puzzle_streak"] == 1
    assert result["records"]["current_puzzle_streak"] == 1


def test_records_repeat_solve_does_not_inflate_streak():
    now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    attempts = [
        _Attempt(1, True, now - timedelta(minutes=3), "a"),
        _Attempt(2, True, now - timedelta(minutes=2), "a"),
        _Attempt(3, True, now - timedelta(minutes=1), "b"),
    ]

    result = build_user_records([], attempts, now=now)

    # Повторное решение уже решённого a серию не удваивает: a + b = 2.
    assert result["records"]["best_puzzle_streak"] == 2
    assert result["records"]["current_puzzle_streak"] == 2
