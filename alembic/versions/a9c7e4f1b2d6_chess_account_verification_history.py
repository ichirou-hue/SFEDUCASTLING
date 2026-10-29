"""chess account verification statuses and history

Revision ID: a9c7e4f1b2d6
Revises: f6a1c2d3e4b5
Create Date: 2026-10-14
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a9c7e4f1b2d6"
down_revision = "f6a1c2d3e4b5"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("user_chess_accounts") as batch_op:
        batch_op.add_column(
            sa.Column("verification_status", sa.String(length=32), server_default="linked", nullable=False)
        )
        batch_op.add_column(sa.Column("verification_method", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("verification_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("verification_failure_reason", sa.String(length=255), nullable=True))

    # Существующие OAuth-подтверждения из v1 автоматически переводим в новую модель доверия.
    op.execute(
        sa.text(
            """
            UPDATE user_chess_accounts
               SET verification_status = 'verified',
                   verification_method = 'oauth',
                   last_verified_at = verified_at
             WHERE verified = true
            """
        )
    )

    op.create_table(
        "chess_account_verifications",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chess_account_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(length=32), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("expected_username", sa.String(length=64), nullable=True),
        sa.Column("external_username", sa.String(length=64), nullable=True),
        sa.Column("external_user_id", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column("details", _JSON, nullable=True),
        sa.ForeignKeyConstraint(["chess_account_id"], ["user_chess_accounts.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chess_account_verifications_account_id",
        "chess_account_verifications",
        ["chess_account_id"],
        unique=False,
    )
    op.create_index(
        "ix_chess_account_verifications_user_id",
        "chess_account_verifications",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chess_account_verifications_user_id",
        table_name="chess_account_verifications",
    )
    op.drop_index(
        "ix_chess_account_verifications_account_id",
        table_name="chess_account_verifications",
    )
    op.drop_table("chess_account_verifications")

    with op.batch_alter_table("user_chess_accounts") as batch_op:
        batch_op.drop_column("verification_failure_reason")
        batch_op.drop_column("verification_expires_at")
        batch_op.drop_column("last_verified_at")
        batch_op.drop_column("verification_method")
        batch_op.drop_column("verification_status")
