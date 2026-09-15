"""make user theme difficulty timestamp timezone-aware

Revision ID: d8b41a2c6f70
Revises: c2f17a8d4e90
Create Date: 2026-09-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8b41a2c6f70"
down_revision: Union[str, None] = "c2f17a8d4e90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "user_theme_difficulties",
        "updated_at",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        existing_nullable=False,
        existing_server_default=sa.text("now()"),
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )


def downgrade() -> None:
    op.alter_column(
        "user_theme_difficulties",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        existing_nullable=False,
        existing_server_default=sa.text("now()"),
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )
