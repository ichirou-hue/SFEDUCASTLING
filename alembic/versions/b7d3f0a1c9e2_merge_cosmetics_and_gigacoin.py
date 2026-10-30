"""merge cosmetics migration and gigacoin/user-activity merge

Revision ID: b7d3f0a1c9e2
Revises: a4c1e9f2b7d6, a4c8d1e6f902
Create Date: 2026-10-30
"""

from collections.abc import Sequence

revision: str = "b7d3f0a1c9e2"
down_revision: tuple[str, str] = ("a4c1e9f2b7d6", "a4c8d1e6f902")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
