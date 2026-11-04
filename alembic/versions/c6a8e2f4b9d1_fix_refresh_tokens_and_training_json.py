"""fix refresh_tokens uniqueness and training JSON types

Revision ID: c6a8e2f4b9d1
Revises: b1e4d7c9a2f3
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c6a8e2f4b9d1"
down_revision: str | None = "b1e4d7c9a2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # На SQLite JSON хранится как текст, а уникальность token_hash уже
    # обеспечивается индексом ix_refresh_tokens_token_hash.
    if op.get_bind().dialect.name != "postgresql":
        return

    # Таблица refresh_tokens исторически получила и UNIQUE CONSTRAINT, и
    # уникальный индекс. Модель описывает только индекс, поэтому лишнее
    # ограничение удаляем, чтобы autogenerate не видел расхождения.
    op.execute(
        "ALTER TABLE refresh_tokens "
        "DROP CONSTRAINT IF EXISTS refresh_tokens_token_hash_key"
    )

    # Миграция e2a6f74d9b31 создала эти колонки как JSON, хотя модель
    # использует JsonType с вариантом JSONB для PostgreSQL.
    op.alter_column(
        "training_tasks",
        "payload",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="payload::jsonb",
        existing_nullable=True,
    )
    op.alter_column(
        "training_attempts",
        "answer",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="answer::jsonb",
        existing_nullable=False,
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    op.alter_column(
        "training_attempts",
        "answer",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.JSON(),
        postgresql_using="answer::json",
        existing_nullable=False,
    )
    op.alter_column(
        "training_tasks",
        "payload",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.JSON(),
        postgresql_using="payload::json",
        existing_nullable=True,
    )
    op.create_unique_constraint(
        "refresh_tokens_token_hash_key",
        "refresh_tokens",
        ["token_hash"],
    )
