"""Тесты контракта прогресса: никаких `topics` и никаких нулевых заглушек accuracy.

B2-аудит зафиксировал два правила:
- `get_training_progress` больше НЕ считает точность по темам (единственный
  источник совмещённого профиля — build_weakness_profile / /api/learning/weaknesses);
- `accuracy` отдаётся как None, когда попыток ещё нет (витрина рисует «—»,
  а не сфабрикованные нули).
"""

import asyncio

from backend.services.training_service import get_training_progress


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Module:
    def __init__(self, id, slug, title):
        self.id = id
        self.slug = slug
        self.title = title


class _FakeDB:
    """Синтаксис-заглушка БД без реального Postgres.

    db.scalars(select(TrainingModule)) -> список модулей.
    db.execute(...)                    -> task_rows, затем attempt_rows.
    """

    def __init__(self, modules, task_rows=(), attempt_rows=()):
        self._modules = modules
        self._task_rows = list(task_rows)
        self._attempt_rows = list(attempt_rows)
        self._task_used = False

    async def scalars(self, stmt):
        return _Result(self._modules)

    async def execute(self, stmt):
        if not self._task_used:
            self._task_used = True
            return _Result(self._task_rows)
        return _Result(self._attempt_rows)


def _run(coro):
    return asyncio.run(coro)


def _progress(attempt_rows=()):
    db = _FakeDB(
        modules=[_Module(1, "pawn", "Пешка"), _Module(2, "rook", "Ладья")],
        task_rows=[(100, 1), (101, 2)],
        attempt_rows=attempt_rows,
    )
    return _run(get_training_progress(db, user_id=1))


def test_progress_omits_topics():
    result = _progress(attempt_rows=[(100, True, None, 1)])

    assert "topics" not in result
    assert set(result) == {"modules", "attempts", "streak"}
    assert result["attempts"]["total"] == 1
    assert result["attempts"]["correct"] == 1
    assert result["attempts"]["accuracy"] == 100.0


def test_progress_accuracy_is_none_without_attempts():
    result = _progress(attempt_rows=[])

    assert result["attempts"]["total"] == 0
    assert result["attempts"]["correct"] == 0
    assert result["attempts"]["accuracy"] is None
    assert result["modules"]["total"] == 2
    assert result["modules"]["completed"] == 0


def test_module_completed_only_when_all_tasks_correct():
    # Задание 100 решено верно, 101 — ни разу. Первый модуль пройден, второй нет.
    result = _progress(attempt_rows=[(100, True, None, 1)])

    assert result["modules"]["completed"] == 1
    assert result["modules"]["percent"] == 50.0
    assert result["modules"]["total"] == 2