"""Чинит server_default now() в training_reviews для SQLite.

Ревизия e4a51c7d92f1 создавала created_at/updated_at с сырым
``sa.text("now()")``. На PostgreSQL это корректно, но у SQLite такой
функции нет ("unknown function: now()"), из-за чего любой INSERT в
training_reviews падал 500-й ошибкой и вкладка «Обучение» не принимала
ответы. Заменяем дефолт на диалектно-осознанный CURRENT_TIMESTAMP.

На PostgreSQL колонки уже используют now(), поэтому миграция там — no-op.

Revision ID: d3e8f1a2b4c6
Revises: a7e1c9d24b63
Create Date: 2026-10-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d3e8f1a2b4c6"
down_revision: Union[str, None] = "a7e1c9d24b63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        # PostgreSQL: now() и так работает, менять нечего.
        return

    # SQLite не умеет ALTER COLUMN ... SET DEFAULT, поэтому batch_alter_table
    # пересоздаёт таблицу, сохранив индексы и констрейнты.
    with op.batch_alter_table("training_reviews") as batch_op:
        batch_op.alter_column(
            "created_at",
            existing_type=sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "updated_at",
            existing_type=sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            existing_nullable=False,
        )


def downgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return

    with op.batch_alter_table("training_reviews") as batch_op:
        batch_op.alter_column(
            "created_at",
            existing_type=sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "updated_at",
            existing_type=sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            existing_nullable=False,
        )
