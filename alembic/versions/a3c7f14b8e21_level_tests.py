"""level_tests: persisted and idempotent level-test results

Revision ID: a3c7f14b8e21
Revises: e2a6f74d9b31
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a3c7f14b8e21"
down_revision: str | None = "e2a6f74d9b31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
# Переносимые типы: см. комментарий в fb22cbd6384b_initial_chat_dataset_games.py

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
_NOW = sa.func.now()


def upgrade() -> None:
    op.create_table(
        "level_tests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "question_ids",
            _JSON,
            nullable=False,
        ),
        sa.Column(
            "answers",
            _JSON,
            nullable=True,
        ),
        sa.Column(
            "score",
            _JSON,
            nullable=True,
        ),
        sa.Column("level", sa.Integer(), nullable=True),
        sa.Column("band", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="started", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_NOW,
            nullable=False,
        ),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_level_tests_user_id"),
        "level_tests",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_level_tests_user_id"), table_name="level_tests")
    op.drop_table("level_tests")
