"""РўРµСЃС‚С‹ РєРѕРЅС‚СЂР°РєС‚Р° РїСЂРѕРіСЂРµСЃСЃР°: РЅРёРєР°РєРёС… `topics` Рё РЅРёРєР°РєРёС… РЅСѓР»РµРІС‹С… Р·Р°РіР»СѓС€РµРє accuracy.

B2-Р°СѓРґРёС‚ Р·Р°С„РёРєСЃРёСЂРѕРІР°Р» РґРІР° РїСЂР°РІРёР»Р°:
- `get_training_progress` Р±РѕР»СЊС€Рµ РќР• СЃС‡РёС‚Р°РµС‚ С‚РѕС‡РЅРѕСЃС‚СЊ РїРѕ С‚РµРјР°Рј (РµРґРёРЅСЃС‚РІРµРЅРЅС‹Р№
  РёСЃС‚РѕС‡РЅРёРє СЃРѕРІРјРµС‰С‘РЅРЅРѕРіРѕ РїСЂРѕС„РёР»СЏ - build_weakness_profile / /api/learning/weaknesses);
- `accuracy` РѕС‚РґР°С‘С‚СЃСЏ РєР°Рє None, РєРѕРіРґР° РїРѕРїС‹С‚РѕРє РµС‰С‘ РЅРµС‚ (РІРёС‚СЂРёРЅР° СЂРёСЃСѓРµС‚ В«-В»,
  Р° РЅРµ СЃС„Р°Р±СЂРёРєРѕРІР°РЅРЅС‹Рµ РЅСѓР»Рё).
"""

import asyncio
from datetime import datetime, time, timedelta

from backend.services.local_time import LOCAL_TZ, local_today
from backend.services.training_service import get_training_progress


def _local_noon(days_ago: int = 0) -> datetime:
    """РџРѕР»РґРµРЅСЊ РјРµСЃС‚РЅРѕРіРѕ РґРЅСЏ: РЅРµ Р·Р°РІРёСЃРёС‚ РѕС‚ РІСЂРµРјРµРЅРё Р·Р°РїСѓСЃРєР° С‚РµСЃС‚РѕРІ."""
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
    """РЎРёРЅС‚Р°РєСЃРёСЃ-Р·Р°РіР»СѓС€РєР° Р‘Р” Р±РµР· СЂРµР°Р»СЊРЅРѕРіРѕ Postgres.

    db.scalars(select(TrainingModule)) -> СЃРїРёСЃРѕРє РјРѕРґСѓР»РµР№.
    db.execute(...)                    -> РїРѕ РїРѕСЂСЏРґРєСѓ: Р·Р°РґР°РЅРёСЏ РјРѕРґСѓР»РµР№
    (task_id, lesson_id, module_id), СѓСЂРѕРєРё РјРѕРґСѓР»РµР№ (lesson_id, module_id),
    РїРѕРїС‹С‚РєРё РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ, Р·Р°С‚РµРј РїРѕРїС‹С‚РєРё РїР°Р·Р·Р»РѕРІ
    ((puzzle_id, correct, created_at) - РґР»СЏ СЃРµСЂРёРё).
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
        modules=[_Module(1, "pawn", "РџРµС€РєР°"), _Module(2, "rook", "Р›Р°РґСЊСЏ")],
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
    # Р—Р°РґР°РЅРёРµ 100 СЂРµС€РµРЅРѕ РІРµСЂРЅРѕ, 101 - РЅРё СЂР°Р·Сѓ. РџРµСЂРІС‹Р№ РјРѕРґСѓР»СЊ РїСЂРѕР№РґРµРЅ, РІС‚РѕСЂРѕР№ РЅРµС‚.
    result = _progress(attempt_rows=[(100, True, None, 1)])

    assert result["modules"]["completed"] == 1
    assert result["modules"]["percent"] == 50.0
    assert result["modules"]["total"] == 2


def test_streak_counts_days_with_three_distinct_puzzles():
    # Р”РµРЅСЊ Р·Р°РЅСЏС‚РёР№ - С‚СЂРё Р РђР—РќР«Р• Р·Р°РґР°С‡Рё. Р’С‡РµСЂР° Рё СЃРµРіРѕРґРЅСЏ РїРѕ С‚СЂРё - СЃРµСЂРёСЏ 2.
    now = _local_noon()
    yesterday = _local_noon(days_ago=1)
    db = _FakeDB(
        modules=[_Module(1, "pawn", "РџРµС€РєР°"), _Module(2, "rook", "Р›Р°РґСЊСЏ")],
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
    # РЎС‡С‘С‚С‡РёРєРё РєСѓСЂСЃР° РїСЂРё СЌС‚РѕРј РЅРµ РјРµРЅСЏСЋС‚СЃСЏ - РїР°Р·Р·Р»С‹ РІР»РёСЏСЋС‚ С‚РѕР»СЊРєРѕ РЅР° СЃРµСЂРёСЋ.
    assert result["attempts"]["total"] == 0
    assert result["modules"]["completed"] == 0


def test_two_puzzles_in_a_day_are_not_enough():
    # Р’С‡РµСЂР° С‚СЂРё Р·Р°РґР°С‡Рё, СЃРµРіРѕРґРЅСЏ С‚РѕР»СЊРєРѕ РґРІРµ - СЃРµРіРѕРґРЅСЏ РµС‰С‘ РЅРµ Р·Р°РЅСЏС‚РёРµ,
    # СЃРµСЂРёСЏ РґРµСЂР¶РёС‚СЃСЏ Р·Р° РІС‡РµСЂР°С€РЅРёР№ РґРµРЅСЊ Рё СЂР°РІРЅР° 1.
    now = _local_noon()
    yesterday = _local_noon(days_ago=1)
    db = _FakeDB(
        modules=[_Module(1, "pawn", "РџРµС€РєР°")],
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
    # РћРґРЅР° Рё С‚Р° Р¶Рµ Р·Р°РґР°С‡Р°, СЂРµС€С‘РЅРЅР°СЏ С‚СЂРёР¶РґС‹, РЅРµ РґРµР»Р°РµС‚ РґРµРЅСЊ Р·Р°РЅСЏС‚РёРµРј.
    now = _local_noon()
    db = _FakeDB(
        modules=[_Module(1, "pawn", "РџРµС€РєР°")],
        puzzle_rows=[("same", True, now), ("same", True, now), ("same", True, now)],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 0


def test_visits_do_not_extend_streak():
    # РџСЂРѕСЃС‚РѕР№ Р·Р°С…РѕРґ РЅР° СЃР°Р№С‚ Р±РѕР»СЊС€Рµ РЅРµ РїСЂРѕРґР»РµРІР°РµС‚ СЃРµСЂРёСЋ: СѓС‡РёС‚С‹РІР°СЋС‚СЃСЏ С‚РѕР»СЊРєРѕ
    # РґРЅРё Р·Р°РЅСЏС‚РёР№ (3+ СЂР°Р·РЅС‹С… РїР°Р·Р»Р° РёР»Рё Р·Р°РґР°РЅРёРµ РєСѓСЂСЃР°).
    db = _FakeDB(
        modules=[_Module(1, "pawn", "РџРµС€РєР°")],
        attempt_rows=[],
        puzzle_rows=[],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 0


def test_course_task_counts_as_active_day():
    # Р”РѕСЃС‚Р°С‚РѕС‡РЅРѕ РѕРґРЅРѕРіРѕ Р·Р°РґР°РЅРёСЏ РјРѕРґСѓР»СЏ РєСѓСЂСЃР° Р·Р° РґРµРЅСЊ.
    now = _local_noon()
    db = _FakeDB(
        modules=[_Module(1, "pawn", "РџРµС€РєР°")],
        task_rows=[(100, 10, 1)],
        lesson_rows=[(10, 1)],
        attempt_rows=[(100, True, now, 1)],
    )
    result = _run(get_training_progress(db, user_id=1))

    assert result["streak"] == 1


def test_streak_joins_course_and_puzzle_days():
    # Р’С‡РµСЂР° - С‚СЂРё РїР°Р·Р»Р°, СЃРµРіРѕРґРЅСЏ - Р·Р°РґР°РЅРёРµ РєСѓСЂСЃР°: РґРЅРё РѕР±СЉРµРґРёРЅСЏСЋС‚СЃСЏ.
    now = _local_noon()
    yesterday = _local_noon(days_ago=1)
    db = _FakeDB(
        modules=[_Module(1, "pawn", "РџРµС€РєР°")],
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
        _Module(1, "pieces", "РЁР°С…РјР°С‚РЅР°СЏ Р°Р·Р±СѓРєР°"),
        _Module(2, "endgames", "Р­РЅРґС€РїРёР»СЊ"),
        _Module(3, "openings", "Р”РµР±СЋС‚"),
        _Module(4, "middlegame", "РњРёС‚С‚РµР»СЊС€РїРёР»СЊ: СЃС‚СЂР°С‚РµРіРёСЏ"),
    ]


def test_locked_modules_excluded_from_course_goal():
    # Р‘РµР· Elo С†РµР»СЊ РєСѓСЂСЃР° - С‚РѕР»СЊРєРѕ Р±Р°Р·РѕРІС‹Рµ РјРѕРґСѓР»Рё, РёРЅР°С‡Рµ 100% РЅРµРґРѕСЃС‚РёР¶РёРјС‹.
    # Р—Р°РіР»СѓС€РєР° Р‘Р” РЅРµ СѓРјРµРµС‚ С„РёР»СЊС‚СЂРѕРІР°С‚СЊ РїРѕ module_ids, РїРѕСЌС‚РѕРјСѓ Р·Р°РґР°РЅРёСЏ Рё
    # СѓСЂРѕРєРё Р·Р°РєСЂС‹С‚С‹С… РјРѕРґСѓР»РµР№ РІ РЅРµС‘ РЅРµ РєР»Р°РґС‘Рј - РєР°Рє Рё РІ СЂРµР°Р»СЊРЅРѕРј SQL.
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
    # РЎ Elo 1000 РјРѕРґСѓР»Рё 1000-1500 РІС…РѕРґСЏС‚ РІ С†РµР»СЊ РєСѓСЂСЃР°.
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
    # РСЃС‚РѕСЂРёСЏ РїРѕРїС‹С‚РѕРє - СЃС‚Р°С‚РёСЃС‚РёРєР°, Р° РЅРµ РґРѕСЃС‚СѓРї: РѕРЅР° РЅРµ РѕР±СЂРµР·Р°РµС‚СЃСЏ РїРѕ Elo.
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
