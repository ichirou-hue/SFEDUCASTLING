"""Единый календарный день приложения (Europe/Moscow).

Все «дневные» метрики (серия дней, дневные лимиты, графики активности)
считаются в одном часовом поясе. Без этого даты, записанные через
``now()``/``current_date()`` в БД, и ``date.today()`` в Python могут
разойтись: игра после полуночи по Москве уезжает во «вчера» по UTC,
и серия дней растёт на единицу «из ниоткуда».

Хранимые отметки времени — naive (``timestamp without time zone``) и
пишутся в UTC: это зафиксировано настройкой сессии БД
(``DATABASE_TIMEZONE``, по умолчанию ``UTC``), см. ``backend/db/session.py``.
Поэтому здесь naive-значение трактуется как UTC и переводится в местную зону.
"""

from datetime import UTC, date, datetime, timedelta, timezone

from backend.config.settings import settings

# Moscow has observed a fixed UTC+3 since 2014. Используем запасной вариант,
# если пакета tzdata нет (частая ситуация на Windows): смещение всё равно верное.
_FALLBACK = timezone(timedelta(hours=3), name="MSK")


def _resolve_timezone():
    name = (settings.timezone or "Europe/Moscow").strip() or "Europe/Moscow"
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return _FALLBACK


LOCAL_TZ = _resolve_timezone()


def to_local_date(value: datetime) -> date:
    """Переводит отметку времени из БД в местную календарную дату."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(LOCAL_TZ).date()


def local_today() -> date:
    """Текущая местная дата (а не серверная и не UTC)."""
    return datetime.now(UTC).astimezone(LOCAL_TZ).date()
