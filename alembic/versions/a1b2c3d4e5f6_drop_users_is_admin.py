"""Удаляем колонку users.is_admin: единственный источник прав — role.

Повышение role IS NULL/не-admin не происходит здесь: прошлая миграция
e5f4b3a2c1d0 уже выставила role='admin' где is_admin=TRUE. Текущая только
убирает продублированную колонку.
"""

import sqlalchemy as sa
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "e5f4b3a2c1d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("users", "is_admin")


def downgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_admin",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Восстановление: админ = роль admin.
    op.execute("UPDATE users SET is_admin = true WHERE role = 'admin'")