"""Align the chess verification index name with SQLAlchemy metadata.

Revision ID: e8f2a4c6d0b1
Revises: c3f8a5d7e2b1
Create Date: 2026-10-31
"""

from collections.abc import Sequence

from alembic import op

revision: str = "e8f2a4c6d0b1"
down_revision: str | None = "c3f8a5d7e2b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index(
        "ix_chess_account_verifications_account_id",
        table_name="chess_account_verifications",
    )
    op.create_index(
        "ix_chess_account_verifications_chess_account_id",
        "chess_account_verifications",
        ["chess_account_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chess_account_verifications_chess_account_id",
        table_name="chess_account_verifications",
    )
    op.create_index(
        "ix_chess_account_verifications_account_id",
        "chess_account_verifications",
        ["chess_account_id"],
    )
