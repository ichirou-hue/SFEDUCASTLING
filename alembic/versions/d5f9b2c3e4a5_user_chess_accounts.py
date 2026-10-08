"""user chess accounts soft links

Revision ID: d5f9b2c3e4a5
Revises: c4e8a1b2d3f4
Create Date: 2026-10-05
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "d5f9b2c3e4a5"
down_revision = "c4e8a1b2d3f4"
branch_labels = None
depends_on = None

# Переносимый тип: PostgreSQL — JSONB, SQLite — JSON (см. fb22cbd6384b).
_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "user_chess_accounts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(length=32), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("rating_type", sa.String(length=32), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("rating_scale", sa.String(length=64), nullable=True),
        sa.Column("games", sa.Integer(), server_default="0", nullable=False),
        sa.Column("rating_deviation", sa.Float(), nullable=True),
        sa.Column("provisional", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("rating_usable", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("profile_snapshot", _JSON, nullable=True),
        sa.Column("linked_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "platform", name="uq_user_chess_accounts_user_platform"),
    )
    op.create_index(
        "ix_user_chess_accounts_user_id",
        "user_chess_accounts",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_user_chess_accounts_user_id", table_name="user_chess_accounts")
    op.drop_table("user_chess_accounts")
