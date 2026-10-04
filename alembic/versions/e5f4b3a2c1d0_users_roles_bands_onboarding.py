"""users: роли + поля методички оценки уровня (role, skill_band, prior_band, внешний рейтинг, онбординг)

Revision ID: e5f4b3a2c1d0
Revises: d8b41a2c6f70
Create Date: 2026-10-04

- role: 'learner' по умолчанию, существующие is_admin=TRUE наследуются в 'admin'.
- skill_band 0–4 (входной тест), prior_band (онбординг), rating_estimate/rating_scale
  (внешний рейтинг в шкале Lichess blitz), onboarding (JSONB сырых ответов Q1–Q8),
  parental_consent (согласие родителя для младше 13).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e5f4b3a2c1d0"
down_revision: str | None = "d8b41a2c6f70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("role", sa.String(length=16), server_default="learner", nullable=False),
    )
    op.add_column("users", sa.Column("skill_band", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("prior_band", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("rating_estimate", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("rating_scale", sa.String(length=32), nullable=True))
    op.add_column(
        "users",
        sa.Column("onboarding", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("users", sa.Column("parental_consent", sa.Boolean(), nullable=True))

    # Наследуем роль администратора из старого булевого флага.
    op.execute("UPDATE users SET role = 'admin' WHERE is_admin = TRUE")


def downgrade() -> None:
    op.drop_column("users", "parental_consent")
    op.drop_column("users", "onboarding")
    op.drop_column("users", "rating_scale")
    op.drop_column("users", "rating_estimate")
    op.drop_column("users", "prior_band")
    op.drop_column("users", "skill_band")
    op.drop_column("users", "role")