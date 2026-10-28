"""user activity days for streak

Revision ID: d9e2f4a7b1c3
Revises: c1d2e3f4a5b6
Create Date: 2026-10-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d9e2f4a7b1c3"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Переносимые типы: см. комментарий в b4d8e2c91f30_puzzle_attempts.py.
_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "user_activity_days",
        sa.Column("id", _ID, autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("activity_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "activity_date",
            name="uq_user_activity_days_user_date",
        ),
    )
    op.create_index(
        "ix_user_activity_days_user_id",
        "user_activity_days",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_activity_days_date",
        "user_activity_days",
        ["activity_date"],
        unique=False,
    )

    # День регистрации существующих аккаунтов считаем первым визитом,
    # чтобы серия не начиналась с нуля задним числом.
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        op.execute(
            "INSERT INTO user_activity_days (user_id, activity_date) "
            "SELECT id, DATE(created_at) FROM users"
        )
    else:
        op.execute(
            "INSERT INTO user_activity_days (user_id, activity_date) "
            "SELECT id, created_at::date FROM users"
        )


def downgrade() -> None:
    op.drop_index("ix_user_activity_days_date", table_name="user_activity_days")
    op.drop_index("ix_user_activity_days_user_id", table_name="user_activity_days")
    op.drop_table("user_activity_days")
