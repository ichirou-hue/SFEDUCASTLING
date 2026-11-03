"""Merge GigaCoin, activity-day, and account verification migration heads.

Revision ID: c3f8a5d7e2b1
Revises: a4c8d1e6f902, b7d4e9a3c1f0, d9e2f4a7b1c3
Create Date: 2026-10-31
"""

from collections.abc import Sequence

revision: str = "c3f8a5d7e2b1"
down_revision: tuple[str, str, str] = (
    "a4c8d1e6f902",
    "b7d4e9a3c1f0",
    "d9e2f4a7b1c3",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
