"""games: удаляем неиспользуемую строковую колонку date

Её роль давно выполняет created_at (timestamp). Отдельная
колонка date varchar(32) нигде не читается и не пишется —
мёртвый груз эпохи PGN-парсера.

Revision ID: d7c9e4b1a2f3
Revises: b8e4f2a6c9d1
Create Date: 2026-08-24 11:00:00

"""

import sqlalchemy as sa
from alembic import op

revision = "d7c9e4b1a2f3"
down_revision = "b8e4f2a6c9d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("games", "date")


def downgrade() -> None:
    op.add_column(
        "games",
        sa.Column("date", sa.String(length=32), nullable=True),
    )
