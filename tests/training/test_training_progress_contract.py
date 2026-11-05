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
from backend.services.training_service import get_training_progress


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
    def __init__(self, id, slug, title):
        self.id = id
        self.slug = slug
        self.title = title


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
        return _Result(self._modules)

    async def execute(self, stmt):
        rows = self._exec_results.pop(0) if self._exec_results else []
        return _Result(rows)


def _run(coro):
    return asyncio.run(coro)


def _progress(attempt_rows=()):
    db = _FakeDB(
        modules=[_Module(1, "pawn", "Пешка"), _Module(2, "rook", "Ладья")],
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
        modules=[_Module(1, "pawn", "Пешка"), _Module(2, "rook", "Ладья")],
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
        modules=[_Module(1, "pawn", "Пешка")],
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
        modules=[_Module(1, "pawn", "Пешка")],
        puzzle_rows=[("same", True, now), ("same", True, now), ("same", True, now)],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 0


def test_visits_do_not_extend_streak():
    # Простой заход на сайт больше не продлевает серию: учитываются только
    # дни занятий (3+ разных пазла или задание курса).
    db = _FakeDB(
        modules=[_Module(1, "pawn", "Пешка")],
        attempt_rows=[],
        puzzle_rows=[],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 0


def test_course_task_counts_as_active_day():
    # Достаточно одного задания модуля курса за день.
    now = _local_noon()
    db = _FakeDB(
        modules=[_Module(1, "pawn", "Пешка")],
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
        modules=[_Module(1, "pawn", "Пешка")],
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


def _course_modules():
    return [
        _Module(1, "pieces", "Шахматная азбука"),
        _Module(2, "endgames", "Эндшпиль"),
        _Module(3, "openings", "Дебют"),
        _Module(4, "middlegame", "Миттельшпиль: стратегия"),
    ]


def test_locked_modules_excluded_from_course_goal():
    # Без Elo цель курса - только базовые модули, иначе 100% недостижимы.
    # Заглушка БД не умеет фильтровать по module_ids, поэтому задания и
    # уроки закрытых модулей в неё не кладём - как и в реальном SQL.
    db = _FakeDB(
        modules=_course_modules(),
        task_rows=[(100, 10, 1)],
        lesson_rows=[(10, 1)],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["modules"]["total"] == 1
    assert [item["slug"] for item in result["modules"]["items"]] == ["pieces"]
    assert result["modules"]["completed"] == 0
    assert result["modules"]["percent"] == 0.0


def test_locked_modules_join_goal_when_elo_is_enough():
    # С Elo 1000 модули 1000-1500 входят в цель курса.
    db = _FakeDB(
        modules=_course_modules(),
        task_rows=[(100, 10, 1), (101, 20, 2), (102, 30, 3), (103, 40, 4)],
        lesson_rows=[(10, 1), (20, 2), (30, 3), (40, 4)],
    )
    result = _run(get_training_progress(db, user_id=1, user_elo=1000))

    assert result["modules"]["total"] == 4
    assert {item["slug"] for item in result["modules"]["items"]} == {
        "pieces",
        "endgames",
        "openings",
        "middlegame",
    }


def test_progress_counts_attempts_from_locked_modules():
    # История попыток - статистика, а не доступ: она не обрезается по Elo.
    db = _FakeDB(
        modules=_course_modules()[:2],
        task_rows=[(100, 10, 1)],
        lesson_rows=[(10, 1)],
        attempt_rows=[(101, True, _local_noon(), 2)],
    )
    result = _run(get_training_progress(db, user_id=1, user_elo=None))

    assert result["modules"]["total"] == 1
    assert result["attempts"]["total"] == 1
    assert result["attempts"]["correct"] == 1
    assert result["attempts"]["accuracy"] == 100.0
