"""puzzle attempts hint_used

Revision ID: fc20095be1de
Revises: b1e7c4a9d2f6
Create Date: 2026-11-05 17:35:01.738435
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "fc20095be1de"
down_revision: str | None = "b1e7c4a9d2f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Факт использования подсказки при решении пазла. Решение с подсказкой
    # засчитывается как решённое, но сбрасывает серию (стрик) в 0.
    op.add_column(
        "puzzle_attempts",
        sa.Column("hint_used", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("puzzle_attempts", "hint_used")
