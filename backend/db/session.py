"""Асинхронный движок и фабрика сессий SQLAlchemy для PostgreSQL.

Подключение настраивается через группу ``database`` в settings.py
(переменная окружения ``DATABASE_URL``).
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.config.settings import settings


def _connect_args() -> dict:
    """Параметры соединения, зависящие от драйвера.

    Для PostgreSQL фиксируем зону сессии (``DATABASE_TIMEZONE``, по умолчанию
    UTC). Колонки — ``timestamp without time zone``, поэтому ``now()`` пишет
    именно «настенное» время этой зоны. Если не зафиксировать, оно зависит от
    настроек конкретного сервера БД, и календарный день в базе расходится с
    приложением (серия дней «прыгает» возле полуночи).
    """
    url = settings.database.url
    if url.startswith("postgresql"):
        return {"server_settings": {"timezone": settings.database.timezone}}
    return {}


# NullPool: каждое соединение живёт один запрос. Это позволяет одному
# движку работать из разных event loop (в т.ч. в pytest с несколькими
# TestClient), не конфликтуя за переиспользование соединений asyncpg.
engine = create_async_engine(
    settings.database.url,
    echo=settings.database.echo,
    poolclass=NullPool,
    connect_args=_connect_args(),
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI-зависимость: выдаёт асинхронную сессию БД на запрос."""
    async with async_session_factory() as session:
        yield session
