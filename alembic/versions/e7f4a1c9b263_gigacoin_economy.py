"""Add GigaCoin balance and idempotent reward ledger.

Revision ID: e7f4a1c9b263
Revises: c1d2e3f4a5b6
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e7f4a1c9b263"
down_revision: str | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("gigacoins", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_table(
        "gigacoin_events",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("ref", sa.String(length=128), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "reason", "ref", name="uq_gigacoin_event_user_reason_ref"),
    )
    op.create_index("ix_gigacoin_events_user_id", "gigacoin_events", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_gigacoin_events_user_id", table_name="gigacoin_events")
    op.drop_table("gigacoin_events")
    op.drop_column("users", "gigacoins")
