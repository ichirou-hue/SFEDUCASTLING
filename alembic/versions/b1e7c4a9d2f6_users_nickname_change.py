"""Add users.nickname and GigaCoin nickname-change state.

Revision ID: b1e7c4a9d2f6
Revises: 3f7c1a9d5e20
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b1e7c4a9d2f6"
down_revision: str | None = "3f7c1a9d5e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # nickname nullable: у записей, созданных в обход API, допускается
    # fallback на login (см. User.public()). Для существующих пользователей
    # значение заполняется логином, чтобы никнейм сразу отображался.
    op.add_column("users", sa.Column("nickname", sa.String(length=32), nullable=True))
    op.add_column(
        "users",
        sa.Column("nickname_changes", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "users",
        sa.Column("nickname_changed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(sa.text("UPDATE users SET nickname = login WHERE nickname IS NULL"))
    op.create_index("ix_users_nickname", "users", ["nickname"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_nickname", table_name="users")
    op.drop_column("users", "nickname_changed_at")
    op.drop_column("users", "nickname_changes")
    op.drop_column("users", "nickname")
