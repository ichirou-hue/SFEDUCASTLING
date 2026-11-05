"""Store server-generated avatar filename on users.

Revision ID: b6e2c9f4a1d7
Revises: 3f7c1a9d5e20
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6e2c9f4a1d7"
down_revision: str | None = "3f7c1a9d5e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("avatar_filename", sa.String(length=80), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "avatar_filename")
