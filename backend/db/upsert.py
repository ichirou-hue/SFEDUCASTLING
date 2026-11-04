"""Диалект-независимый INSERT ... ON CONFLICT.

Приложение по умолчанию работает на SQLite, но ``pg_insert`` из диалекта
PostgreSQL на нём падает. Выбираем реализацию по фактическому подключению.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession


def dialect_insert(db: AsyncSession, model: Any):
    """Возвращает insert() нужного диалекта с поддержкой on_conflict_*."""
    if db.get_bind().dialect.name == "sqlite":
        return sqlite_insert(model)
    return pg_insert(model)
