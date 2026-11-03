"""Persist puzzle streak progress and purchased cosmetics.

Revision ID: a4c8d1e6f902
Revises: f9a7b2c1d6e3
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4c8d1e6f902"
down_revision: str | None = "f9a7b2c1d6e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("puzzle_streak", sa.Integer(), server_default="0", nullable=False))
    op.add_column("users", sa.Column("equipped_avatar_frame", sa.String(length=64), nullable=True))
    op.add_column("users", sa.Column("equipped_profile_accent", sa.String(length=64), nullable=True))
    op.create_table(
        "user_cosmetics",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("cosmetic_id", sa.String(length=64), nullable=False),
        sa.Column("is_equipped", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("purchased_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "cosmetic_id", name="uq_user_cosmetic"),
    )
    op.create_index("ix_user_cosmetics_user_id", "user_cosmetics", ["user_id"])

    connection = op.get_bind()
    users = sa.table("users", sa.column("id", sa.Integer), sa.column("puzzle_streak", sa.Integer))
    attempts = sa.table(
        "puzzle_attempts",
        sa.column("id", sa.BigInteger),
        sa.column("user_id", sa.Integer),
        sa.column("correct", sa.Boolean),
    )
    streaks: dict[int, int] = {}
    broken: set[int] = set()
    history = connection.execute(
        sa.select(attempts.c.user_id, attempts.c.correct)
        .order_by(attempts.c.user_id, attempts.c.id.desc())
    )
    for attempt in history:
        user_id = int(attempt.user_id)
        if user_id in broken:
            continue
        if attempt.correct:
            streaks[user_id] = streaks.get(user_id, 0) + 1
        else:
            broken.add(user_id)
    for user_id, streak in streaks.items():
        connection.execute(
            sa.update(users).where(users.c.id == user_id).values(puzzle_streak=streak)
        )


def downgrade() -> None:
    op.drop_index("ix_user_cosmetics_user_id", table_name="user_cosmetics")
    op.drop_table("user_cosmetics")
    op.drop_column("users", "equipped_profile_accent")
    op.drop_column("users", "equipped_avatar_frame")
    op.drop_column("users", "puzzle_streak")
