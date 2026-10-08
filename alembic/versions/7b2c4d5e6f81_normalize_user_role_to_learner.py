"""Normalize ordinary user role to learner.

Revision ID: 7b2c4d5e6f81
Revises: 519591271e20
Create Date: 2026-10-04

The project historically uses ``learner`` for an ordinary user and ``admin``
for an administrator.  A temporary auth patch introduced ``user``.  This
migration restores the original naming and makes ``learner`` the DB default.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7b2c4d5e6f81"
down_revision: str | None = "519591271e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE users SET role = 'learner' WHERE role = 'user'")
    # batch_alter_table: SQLite не умеет ALTER COLUMN — таблица пересоздаётся,
    # на PostgreSQL batch выполняет обычные ALTER.
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "role",
            existing_type=sa.String(length=32),
            existing_nullable=False,
            server_default="learner",
        )


def downgrade() -> None:
    op.execute("UPDATE users SET role = 'user' WHERE role = 'learner'")
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "role",
            existing_type=sa.String(length=32),
            existing_nullable=False,
            server_default="user",
        )
