"""training: modules, lessons, tasks, attempts

Revision ID: e2a6f74d9b31
Revises: d7c9e4b1a2f3
Create Date: 2026-09-09

Создаёт базовую структуру отдельной подсистемы «Обучение».
Контент уроков и заданий будет добавляться отдельно, чтобы миграция
схемы БД не была связана с наполнением учебного курса.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e2a6f74d9b31"
down_revision: str | None = "d7c9e4b1a2f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_modules",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_training_modules_slug"),
        "training_modules",
        ["slug"],
        unique=True,
    )

    op.create_table(
        "training_lessons",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("module_id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=128), nullable=False),
        sa.Column("theory", sa.Text(), nullable=True),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.ForeignKeyConstraint(
            ["module_id"],
            ["training_modules.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "module_id",
            "slug",
            name="uq_training_lessons_module_slug",
        ),
    )
    op.create_index(
        op.f("ix_training_lessons_module_id"),
        "training_lessons",
        ["module_id"],
        unique=False,
    )

    op.create_table(
        "training_tasks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("lesson_id", sa.Integer(), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=128), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("fen", sa.Text(), nullable=False),
        sa.Column("source_square", sa.String(length=2), nullable=True),
        sa.Column("difficulty", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.ForeignKeyConstraint(
            ["lesson_id"],
            ["training_lessons.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_training_tasks_lesson_id"),
        "training_tasks",
        ["lesson_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_tasks_task_type"),
        "training_tasks",
        ["task_type"],
        unique=False,
    )

    op.create_table(
        "training_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column(
            "answer",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("correct", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("score", sa.Float(), server_default="0", nullable=False),
        sa.Column("attempt_number", sa.Integer(), server_default="1", nullable=False),
        sa.Column("hints_used", sa.Integer(), server_default="0", nullable=False),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["training_tasks.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_training_attempts_task_id"),
        "training_attempts",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_training_attempts_user_id"),
        "training_attempts",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_training_attempts_user_id"), table_name="training_attempts")
    op.drop_index(op.f("ix_training_attempts_task_id"), table_name="training_attempts")
    op.drop_table("training_attempts")

    op.drop_index(op.f("ix_training_tasks_task_type"), table_name="training_tasks")
    op.drop_index(op.f("ix_training_tasks_lesson_id"), table_name="training_tasks")
    op.drop_table("training_tasks")

    op.drop_index(op.f("ix_training_lessons_module_id"), table_name="training_lessons")
    op.drop_table("training_lessons")

    op.drop_index(op.f("ix_training_modules_slug"), table_name="training_modules")
    op.drop_table("training_modules")
