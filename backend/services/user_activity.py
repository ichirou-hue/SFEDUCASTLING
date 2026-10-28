"""Учёт дней, когда пользователь заходил в приложение.

Визит фиксируется при регистрации, входе, обновлении сессии и открытии
профиля (/api/auth/me). Серия дней строится по объединению этих дат с
днями учебной активности — см. ``training_service._current_streak``.
"""

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.user_activity_day import UserActivityDay


async def record_user_visit(db: AsyncSession, user_id: int) -> None:
    """Отмечает сегодняшний день как день визита (идемпотентно).

    День берём из БД (``current_date()``), чтобы он совпадал с датой,
    которую использует дневной лимит пробных задач. Повторный вызов в
    тот же день ничего не пишет, а гонку параллельных запросов гасит
    ``ON CONFLICT DO NOTHING`` по уникальному индексу
    ``(user_id, activity_date)``.
    """
    today = await db.scalar(select(func.current_date()))
    if today is None:
        return
    if isinstance(today, str):
        today = date.fromisoformat(today)

    values = {"user_id": user_id, "activity_date": today}
    if db.get_bind().dialect.name == "sqlite":
        stmt = sqlite_insert(UserActivityDay).values(**values)
    else:
        stmt = pg_insert(UserActivityDay).values(**values)
    await db.execute(
        stmt.on_conflict_do_nothing(index_elements=["user_id", "activity_date"])
    )
    await db.commit()
