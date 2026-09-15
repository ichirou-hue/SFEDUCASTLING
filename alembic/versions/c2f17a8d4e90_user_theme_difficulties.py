"""dynamic difficulty per user and theme

Revision ID: c2f17a8d4e90
Revises: b4d8e2c91f30
Create Date: 2026-09-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c2f17a8d4e90"
down_revision: Union[str, None] = "b4d8e2c91f30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_theme_difficulties",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("theme_slug", sa.String(length=64), nullable=False),
        sa.Column("current_difficulty", sa.Integer(), server_default="2", nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "current_difficulty >= 1 AND current_difficulty <= 3",
            name="ck_user_theme_difficulty_range",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "theme_slug", name="uq_user_theme_difficulty"),
    )
    op.create_index(
        "ix_user_theme_difficulties_user_id",
        "user_theme_difficulties",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_user_theme_difficulties_theme_slug",
        "user_theme_difficulties",
        ["theme_slug"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_theme_difficulties_theme_slug",
        table_name="user_theme_difficulties",
    )
    op.drop_index(
        "ix_user_theme_difficulties_user_id",
        table_name="user_theme_difficulties",
    )
    op.drop_table("user_theme_difficulties")
