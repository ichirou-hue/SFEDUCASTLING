"""training lessons section title

Revision ID: c1d2e3f4a5b6
Revises: f6b1e2a4c8d9
Create Date: 2026-10-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "f6b1e2a4c8d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_lessons",
        sa.Column("section_title", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_lessons", "section_title")
