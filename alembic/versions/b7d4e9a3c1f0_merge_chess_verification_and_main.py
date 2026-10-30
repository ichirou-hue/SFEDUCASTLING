"""merge chess-account verification branch with current main

Revision ID: b7d4e9a3c1f0
Revises: c1d2e3f4a5b6, a9c7e4f1b2d6
Create Date: 2026-10-29
"""

from typing import Sequence, Union

revision: str = "b7d4e9a3c1f0"
down_revision: Union[str, Sequence[str], None] = (
    "c1d2e3f4a5b6",
    "a9c7e4f1b2d6",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
