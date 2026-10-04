"""spaced repetition reviews for training tasks

Revision ID: e4a51c7d92f1
Revises: d8b41a2c6f70
Create Date: 2026-09-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4a51c7d92f1"
down_revision: Union[str, None] = "d8b41a2c6f70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_reviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("next_review_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ease", sa.Float(), server_default="2.2", nullable=False),
        sa.Column("reps", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("ease >= 1.0", name="ck_training_review_ease"),
        sa.CheckConstraint("reps >= 0", name="ck_training_review_reps"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["task_id"], ["training_tasks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "task_id", name="uq_training_review_user_task"),
    )
    op.create_index(
        "ix_training_reviews_user_id",
        "training_reviews",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_reviews_task_id",
        "training_reviews",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_reviews_next_review_at",
        "training_reviews",
        ["next_review_at"],
        unique=False,
    )
    op.create_index(
        "ix_training_reviews_user_due",
        "training_reviews",
        ["user_id", "next_review_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_training_reviews_user_due", table_name="training_reviews")
    op.drop_index("ix_training_reviews_next_review_at", table_name="training_reviews")
    op.drop_index("ix_training_reviews_task_id", table_name="training_reviews")
    op.drop_index("ix_training_reviews_user_id", table_name="training_reviews")
    op.drop_table("training_reviews")
