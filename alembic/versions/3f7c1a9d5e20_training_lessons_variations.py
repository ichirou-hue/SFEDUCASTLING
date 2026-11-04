"""training lessons variations

Revision ID: 3f7c1a9d5e20
Revises: e8f2a4c6d0b1
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3f7c1a9d5e20"
down_revision: str | None = "e8f2a4c6d0b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.add_column(
        "training_lessons",
        sa.Column("variations", _JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_lessons", "variations")
