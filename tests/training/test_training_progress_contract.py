"""Тесты контракта прогресса: никаких `topics` и никаких нулевых заглушек accuracy.

B2-аудит зафиксировал два правила:
- `get_training_progress` больше НЕ считает точность по темам (единственный
  источник совмещённого профиля - build_weakness_profile / /api/learning/weaknesses);
- `accuracy` отдаётся как None, когда попыток ещё нет (витрина рисует «-»,
  а не сфабрикованные нули).
"""

import asyncio
from datetime import datetime, time, timedelta

from backend.services.local_time import LOCAL_TZ, local_today
from backend.services.training_course_data import MODULES
from backend.services.training_service import (
    COURSE_MODULE_SLUGS,
    get_module_by_slug,
    get_training_progress,
    list_modules,
)


def _local_noon(days_ago: int = 0) -> datetime:
    """Полдень местного дня: не зависит от времени запуска тестов."""
    day = local_today() - timedelta(days=days_ago)
    return datetime.combine(day, time(12, 0), tzinfo=LOCAL_TZ)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Module:
    def __init__(
        self, id, slug, title, *, description=None, sort_order=0, enabled=True
    ):
        self.id = id
        self.slug = slug
        self.title = title
        self.description = description
        self.sort_order = sort_order
        self.enabled = enabled


class _FakeDB:
    """Синтаксис-заглушка БД без реального Postgres.

    db.scalars(select(TrainingModule)) -> список модулей.
    db.execute(...)                    -> по порядку: задания модулей
    (task_id, lesson_id, module_id), уроки модулей (lesson_id, module_id),
    попытки пользователя, затем попытки паззлов
    ((puzzle_id, correct, created_at) - для серии).
    """

    def __init__(self, modules, task_rows=(), lesson_rows=(), attempt_rows=(),
                 puzzle_rows=()):
        self._modules = modules
        self._exec_results = [
            list(task_rows),
            list(lesson_rows),
            list(attempt_rows),
            list(puzzle_rows),
        ]

    async def scalars(self, stmt):
        modules = self._modules
        if "slug IN" in str(stmt):
            modules = [m for m in modules if m.slug in COURSE_MODULE_SLUGS]
        return _Result(modules)

    async def execute(self, stmt):
        rows = self._exec_results.pop(0) if self._exec_results else []
        return _Result(rows)


def _run(coro):
    return asyncio.run(coro)


def test_module_api_excludes_legacy_database_rows():
    modules = [
        _Module(
            i,
            data["slug"],
            data["title"],
            description=data.get("description"),
            sort_order=data.get("sort_order", 99),
            enabled=data.get("enabled", True),
        )
        for i, data in enumerate(
            [*MODULES, {"slug": "legacy-pawn", "title": "Старый модуль пешки"}],
            start=1,
        )
    ]

    class _ModulesDB:
        async def scalars(self, stmt):
            assert "training_modules.slug IN" in str(stmt)
            return _Result([m for m in modules if m.slug in COURSE_MODULE_SLUGS])

        async def scalar(self, stmt):
            return 0

    result = _run(list_modules(_ModulesDB()))
    assert [module["slug"] for module in result] == [
        module["slug"] for module in MODULES
    ]


def test_legacy_module_cannot_be_opened_by_slug():
    class _NoQueryDB:
        async def scalar(self, stmt):
            raise AssertionError("legacy slug should be rejected before querying DB")

    assert _run(get_module_by_slug(_NoQueryDB(), "legacy-pawn")) is None


def _progress(attempt_rows=()):
    db = _FakeDB(
        modules=[
            _Module(1, "pieces", "Шахматная азбука"),
            _Module(2, "tactics", "Тактические приёмы"),
            _Module(3, "legacy-pawn", "Старый модуль пешки"),
        ],
        task_rows=[(100, 10, 1), (101, 20, 2)],
        lesson_rows=[(10, 1), (20, 2)],
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
    # Задание 100 решено верно, 101 - ни разу. Первый модуль пройден, второй нет.
    result = _progress(attempt_rows=[(100, True, None, 1)])

    assert result["modules"]["completed"] == 1
    assert result["modules"]["percent"] == 50.0
    assert result["modules"]["total"] == 2


def test_streak_counts_days_with_three_distinct_puzzles():
    # День занятий - три РАЗНЫЕ задачи. Вчера и сегодня по три - серия 2.
    now = _local_noon()
    yesterday = _local_noon(days_ago=1)
    db = _FakeDB(
        modules=[
            _Module(1, "pieces", "Шахматная азбука"),
            _Module(2, "tactics", "Тактические приёмы"),
        ],
        task_rows=[(100, 10, 1), (101, 20, 2)],
        lesson_rows=[(10, 1), (20, 2)],
        attempt_rows=[],
        puzzle_rows=[
            ("a1", True, yesterday),
            ("a2", True, yesterday),
            ("a3", True, yesterday),
            ("b1", True, now),
            ("b2", True, now),
            ("b3", True, now),
        ],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 2
    # Счётчики курса при этом не меняются - паззлы влияют только на серию.
    assert result["attempts"]["total"] == 0
    assert result["modules"]["completed"] == 0


def test_two_puzzles_in_a_day_are_not_enough():
    # Вчера три задачи, сегодня только две - сегодня ещё не занятие,
    # серия держится за вчерашний день и равна 1.
    now = _local_noon()
    yesterday = _local_noon(days_ago=1)
    db = _FakeDB(
        modules=[_Module(1, "pieces", "Шахматная азбука")],
        puzzle_rows=[
            ("a1", True, yesterday),
            ("a2", True, yesterday),
            ("a3", True, yesterday),
            ("b1", True, now),
            ("b2", True, now),
        ],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 1


def test_repeated_puzzle_counts_once_per_day():
    # Одна и та же задача, решённая трижды, не делает день занятием.
    now = _local_noon()
    db = _FakeDB(
        modules=[_Module(1, "pieces", "Шахматная азбука")],
        puzzle_rows=[("same", True, now), ("same", True, now), ("same", True, now)],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 0


def test_visits_do_not_extend_streak():
    # Простой заход на сайт больше не продлевает серию: учитываются только
    # дни занятий (3+ разных пазла или задание курса).
    db = _FakeDB(
        modules=[_Module(1, "pieces", "Шахматная азбука")],
        attempt_rows=[],
        puzzle_rows=[],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 0


def test_course_task_counts_as_active_day():
    # Достаточно одного задания модуля курса за день.
    now = _local_noon()
    db = _FakeDB(
        modules=[_Module(1, "pieces", "Шахматная азбука")],
        task_rows=[(100, 10, 1)],
        lesson_rows=[(10, 1)],
        attempt_rows=[(100, True, now, 1)],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 1


def test_streak_joins_course_and_puzzle_days():
    # Вчера - три пазла, сегодня - задание курса: дни объединяются.
    now = _local_noon()
    yesterday = _local_noon(days_ago=1)
    db = _FakeDB(
        modules=[_Module(1, "pieces", "Шахматная азбука")],
        task_rows=[(100, 10, 1)],
        lesson_rows=[(10, 1)],
        attempt_rows=[(100, True, now, 1)],
        puzzle_rows=[
            ("a1", True, yesterday),
            ("a2", True, yesterday),
            ("a3", True, yesterday),
        ],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 2
