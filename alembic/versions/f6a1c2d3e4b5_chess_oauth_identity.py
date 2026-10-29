"""verified chess OAuth identity

Revision ID: f6a1c2d3e4b5
Revises: d3e8f1a2b4c6
Create Date: 2026-10-14
"""

import sqlalchemy as sa
from alembic import op

revision = "f6a1c2d3e4b5"
down_revision = "d3e8f1a2b4c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table корректно работает и с SQLite, где ALTER TABLE ограничен.
    with op.batch_alter_table("user_chess_accounts") as batch_op:
        batch_op.add_column(sa.Column("external_user_id", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_unique_constraint(
            "uq_user_chess_accounts_platform_external_user_id",
            ["platform", "external_user_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("user_chess_accounts") as batch_op:
        batch_op.drop_constraint(
            "uq_user_chess_accounts_platform_external_user_id",
            type_="unique",
        )
        batch_op.drop_column("verified_at")
        batch_op.drop_column("external_user_id")
