"""users.onboarding: сырые ответы онбординг-анкеты Q1-Q8

Revision ID: 7ab260ccce8a
Revises: d8b41a2c6f70
Create Date: 2026-10-03

JSONB сырых ответов анкеты (см. «Анкета оценки уровня игры»,
раздел 1). Пусто (NULL), пока пользователь анкету не заполнил —
по этому же признаку фронтенд показывает/прячет плашку.

TODO(BACKEND): следующая миграция по методике — разложить ответы
по отдельным колонкам users:
    prior_band        int  1..4   (сейчас считается и лежит в JSON)
    rating_estimate   int         шкала Lichess blitz
    rating_scale      str         происхождение шкалы (lichess_blitz и т.п.)
    pedagogy          str         ответ Q6 (подсказки)
    skill_band        int  0..4   пишет входной тест, не эта анкета
Плюс отдельный флаг parental consent (PII и детский режим, раздел 8).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7ab260ccce8a"
down_revision: str | None = "d8b41a2c6f70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Переносимый тип: PostgreSQL — JSONB, SQLite — JSON (см. fb22cbd6384b).
_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    return {col["name"] for col in sa.inspect(bind).get_columns(table_name)}


def upgrade() -> None:
    # Колонку могла добавить параллельная ветка (e5f4b3a2c1d0 на origin/main).
    if "onboarding" not in _columns("users"):
        op.add_column("users", sa.Column("onboarding", _JSON, nullable=True))


def downgrade() -> None:
    if "onboarding" in _columns("users"):
        op.drop_column("users", "onboarding")
