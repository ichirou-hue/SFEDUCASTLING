"""make user theme difficulty timestamp timezone-aware

Revision ID: d8b41a2c6f70
Revises: c2f17a8d4e90
Create Date: 2026-09-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8b41a2c6f70"
down_revision: Union[str, None] = "c2f17a8d4e90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _pg_using() -> dict:
    """Приведение значения нужно только PostgreSQL.

    В SQLite колонка пересоздаётся через batch_alter_table, а там
    временная зона фактически не хранится, так что USING не нужен.
    """
    if op.get_bind().dialect.name == "postgresql":
        return {"postgresql_using": "updated_at AT TIME ZONE 'UTC'"}
    return {}


def upgrade() -> None:
    # batch_alter_table нужен для SQLite: он пересоздаёт таблицу вместо
    # ALTER COLUMN ... TYPE, который SQLite не поддерживает. На PostgreSQL
    # batch-режим выполняет обычный ALTER.
    with op.batch_alter_table("user_theme_difficulties") as batch_op:
        batch_op.alter_column(
            "updated_at",
            existing_type=sa.DateTime(timezone=False),
            type_=sa.DateTime(timezone=True),
            existing_nullable=False,
            existing_server_default=sa.func.now(),
            **_pg_using(),
        )


def downgrade() -> None:
    with op.batch_alter_table("user_theme_difficulties") as batch_op:
        batch_op.alter_column(
            "updated_at",
            existing_type=sa.DateTime(timezone=True),
            type_=sa.DateTime(timezone=False),
            existing_nullable=False,
            existing_server_default=sa.func.now(),
            **_pg_using(),
        )
