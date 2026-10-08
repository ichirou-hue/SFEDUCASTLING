"""Слияние цепочек анкеты (7ab260ccce8a) и ролей/оценки (d5f9b2c3e4a5)

Revision ID: cfd7967d8c5c
Revises: 7ab260ccce8a, d5f9b2c3e4a5
Create Date: 2026-10-07

После слияния веток feature/onboarding-anketa и main у миграций
оказались две головы. Эта ревизия объединяет их в одну.
"""

from collections.abc import Sequence

revision: str = "cfd7967d8c5c"
down_revision: str | Sequence[str] | None = ("7ab260ccce8a", "d5f9b2c3e4a5")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
