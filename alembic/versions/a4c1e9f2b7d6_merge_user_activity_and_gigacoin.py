"""merge user activity days and gigacoin migrations

Revision ID: a4c1e9f2b7d6
Revises: d9e2f4a7b1c3, f9a7b2c1d6e3
Create Date: 2026-10-28
"""

from collections.abc import Sequence

revision: str = "a4c1e9f2b7d6"
down_revision: tuple[str, str] = ("d9e2f4a7b1c3", "f9a7b2c1d6e3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
