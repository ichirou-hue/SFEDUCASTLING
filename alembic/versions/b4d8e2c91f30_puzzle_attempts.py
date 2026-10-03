"""puzzle attempts for learning progress

Revision ID: b4d8e2c91f30
Revises: a3c7f14b8e21
Create Date: 2026-09-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b4d8e2c91f30"
down_revision: Union[str, None] = "a3c7f14b8e21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
# Переносимые типы: см. комментарий в fb22cbd6384b_initial_chat_dataset_games.py

_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_NOW = sa.func.now()


def upgrade() -> None:
    op.create_table(
        "puzzle_attempts",
        sa.Column("id", _ID, autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("puzzle_id", sa.String(length=64), nullable=False),
        sa.Column("correct", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_puzzle_attempts_user_id", "puzzle_attempts", ["user_id"], unique=False)
    op.create_index("ix_puzzle_attempts_puzzle_id", "puzzle_attempts", ["puzzle_id"], unique=False)
    op.create_index(
        "ix_puzzle_attempts_user_puzzle",
        "puzzle_attempts",
        ["user_id", "puzzle_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_puzzle_attempts_user_puzzle", table_name="puzzle_attempts")
    op.drop_index("ix_puzzle_attempts_puzzle_id", table_name="puzzle_attempts")
    op.drop_index("ix_puzzle_attempts_user_id", table_name="puzzle_attempts")
    op.drop_table("puzzle_attempts")
