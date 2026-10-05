"""assessment onboarding and personalized level test

Revision ID: c4e8a1b2d3f4
Revises: 7b2c4d5e6f81
Create Date: 2026-10-04
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c4e8a1b2d3f4"
down_revision = "7b2c4d5e6f81"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    return {col["name"] for col in sa.inspect(bind).get_columns(table_name)}


def upgrade() -> None:
    # e5f4b3a2c1d0 в части установок уже создавала часть этих полей.
    # Поэтому миграция безопасно добавляет только реально отсутствующие.
    user_cols = _columns("users")
    if "skill_band" not in user_cols:
        op.add_column("users", sa.Column("skill_band", sa.Integer(), nullable=True))
    if "prior_band" not in user_cols:
        op.add_column("users", sa.Column("prior_band", sa.Integer(), nullable=True))
    if "rating_estimate" not in user_cols:
        op.add_column("users", sa.Column("rating_estimate", sa.Integer(), nullable=True))
    if "rating_scale" not in user_cols:
        op.add_column("users", sa.Column("rating_scale", sa.String(length=64), nullable=True))
    if "onboarding" not in user_cols:
        op.add_column("users", sa.Column("onboarding", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    if "assessment_completed_at" not in user_cols:
        op.add_column("users", sa.Column("assessment_completed_at", sa.DateTime(timezone=True), nullable=True))

    test_cols = _columns("level_tests")
    if "seed" not in test_cols:
        op.add_column("level_tests", sa.Column("seed", sa.Integer(), nullable=True))
    if "initial_rating" not in test_cols:
        op.add_column("level_tests", sa.Column("initial_rating", sa.Integer(), nullable=True))
    if "rating_group" not in test_cols:
        op.add_column("level_tests", sa.Column("rating_group", sa.String(length=32), nullable=True))
    if "metrics_snapshot" not in test_cols:
        op.add_column("level_tests", sa.Column("metrics_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    test_cols = _columns("level_tests")
    for name in ("metrics_snapshot", "rating_group", "initial_rating", "seed"):
        if name in test_cols:
            op.drop_column("level_tests", name)

    # Поля skill_band/prior_band/rating_estimate/rating_scale/onboarding могли
    # существовать до этой ревизии, поэтому downgrade их намеренно не удаляет.
    user_cols = _columns("users")
    if "assessment_completed_at" in user_cols:
        op.drop_column("users", "assessment_completed_at")
