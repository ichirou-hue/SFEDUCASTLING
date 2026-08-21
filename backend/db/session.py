"""Асинхронный движок и фабрика сессий SQLAlchemy для PostgreSQL.

Подключение настраивается через группу ``database`` в settings.py
(переменная окружения ``DATABASE_URL``).
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.config.settings import settings

# NullPool: каждое соединение живёт один запрос. Это позволяет одному
# движку работать из разных event loop (в т.ч. в pytest с несколькими
# TestClient), не конфликтуя за переиспользование соединений asyncpg.
engine = create_async_engine(
    settings.database.url,
    echo=settings.database.echo,
    poolclass=NullPool,
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI-зависимость: выдаёт асинхронную сессию БД на запрос."""
    async with async_session_factory() as session:
        yield session
