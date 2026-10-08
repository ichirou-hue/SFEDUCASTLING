"""Базовый класс декларативных SQLAlchemy-моделей и переносимые типы колонок.

Проект должен одинаково работать на PostgreSQL и SQLite, а также
на Linux и Windows. Для этого здесь описаны два «переносимых» типа:

- ``BigIntPK``  — BigInteger-первичный ключ. PostgreSQL получает
  BIGSERIAL (как и раньше), а SQLite — INTEGER, потому что только
  INTEGER PRIMARY KEY умеет автоинкрементиться (иначе INSERT падает
  с "NOT NULL constraint failed: *.id").
- ``JsonType``  — JSON-колонка. PostgreSQL получает JSONB (как и
  раньше, сохраняется тип для продакшена), а SQLite — JSON.

Использовать их нужно вместо ``BigInteger`` и ``postgresql.JSONB``
в моделях. В миграциях (alembic/versions/) типы продублированы
локально: миграции обязаны быть самодостаточными и не импортировать
код приложения, который может измениться.
"""

from sqlalchemy import JSON, BigInteger, Integer
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


BigIntPK = BigInteger().with_variant(Integer(), "sqlite")

JsonType = JSON().with_variant(postgresql.JSONB(), "postgresql")
