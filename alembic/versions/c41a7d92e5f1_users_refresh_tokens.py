"""users + refresh_tokens (задача 64: регистрация и авторизация)

Revision ID: c41a7d92e5f1
Revises: fb22cbd6384b
Create Date: 2026-08-23

users.login — уникальный, users.email — уникальный (опционален).
refresh_tokens.token_hash — sha256 от сырого токена, уникальный.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'c41a7d92e5f1'
down_revision: str | None = 'fb22cbd6384b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('login', sa.String(length=32), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=True),
    sa.Column('password_hash', sa.String(length=128), nullable=False),
    sa.Column('elo', sa.Integer(), nullable=True),
    sa.Column('is_admin', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_login'), 'users', ['login'], unique=True)
    op.create_table('refresh_tokens',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('revoked', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token_hash')
    )
    op.create_index(op.f('ix_refresh_tokens_user_id'), 'refresh_tokens', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_refresh_tokens_user_id'), table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    op.drop_index(op.f('ix_users_login'), table_name='users')
    op.drop_table('users')
