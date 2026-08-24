"""games/dataset_moves/chat_messages: user_id -> FK на users.id

Наводим ссылочную целостность (задача «нормальная БД»):
- games.user_id: varchar -> integer FK users.id (ON DELETE SET NULL)
- dataset_moves.user_id: varchar NOT NULL 'anonymous' -> integer NULL FK
- chat_messages: добавлен user_id integer NULL FK

Старые строковые идентификаторы эпохи без авторизации
('test-int', 'user_xxx', 'anonymous', ...) превращаются в NULL:
анонимные данные сохраняются, но без привязки к аккаунту.

Revision ID: b8e4f2a6c9d1
Revises: c41a7d92e5f1
Create Date: 2026-08-24 00:30:00

"""

from alembic import op
import sqlalchemy as sa

revision = "b8e4f2a6c9d1"
down_revision = "c41a7d92e5f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- games.user_id ---
    # Мусорные строки и ссылки на несуществующих юзеров -> NULL.
    op.execute(
        "UPDATE games SET user_id = NULL "
        "WHERE user_id IS NOT NULL AND user_id !~ '^[0-9]+$'"
    )
    op.execute(
        "UPDATE games g SET user_id = NULL "
        "WHERE g.user_id IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM users u WHERE u.id = g.user_id::integer)"
    )
    op.alter_column(
        "games",
        "user_id",
        existing_type=sa.String(length=64),
        type_=sa.Integer(),
        nullable=True,
        postgresql_using="NULLIF(user_id, '')::integer",
    )
    op.create_foreign_key(
        "fk_games_user_id_users",
        "games",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # --- dataset_moves.user_id ---
    # Порядок критичен: сначала снимаем NOT NULL/default,
    # только потом можно писать NULL в старые строки.
    op.alter_column(
        "dataset_moves",
        "user_id",
        existing_type=sa.String(length=64),
        nullable=True,
        server_default=None,
    )
    op.execute(
        "UPDATE dataset_moves SET user_id = NULL WHERE user_id !~ '^[0-9]+$'"
    )
    op.alter_column(
        "dataset_moves",
        "user_id",
        existing_type=sa.String(length=64),
        type_=sa.Integer(),
        postgresql_using="CASE WHEN user_id ~ '^[0-9]+$' THEN user_id::integer END",
    )
    op.execute(
        "UPDATE dataset_moves d SET user_id = NULL "
        "WHERE d.user_id IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM users u WHERE u.id = d.user_id)"
    )
    op.create_foreign_key(
        "fk_dataset_moves_user_id_users",
        "dataset_moves",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # --- chat_messages.user_id (новая колонка) ---
    op.add_column(
        "chat_messages",
        sa.Column("user_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_chat_messages_user_id",
        "chat_messages",
        ["user_id"],
    )
    op.create_foreign_key(
        "fk_chat_messages_user_id_users",
        "chat_messages",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # --- chat_messages ---
    op.drop_constraint(
        "fk_chat_messages_user_id_users",
        "chat_messages",
        type_="foreignkey",
    )
    op.drop_index("ix_chat_messages_user_id", table_name="chat_messages")
    op.drop_column("chat_messages", "user_id")

    # --- dataset_moves ---
    op.drop_constraint(
        "fk_dataset_moves_user_id_users",
        "dataset_moves",
        type_="foreignkey",
    )
    op.alter_column(
        "dataset_moves",
        "user_id",
        existing_type=sa.Integer(),
        type_=sa.String(length=64),
        nullable=False,
        server_default=sa.text("'anonymous'"),
        postgresql_using="COALESCE(user_id::text, 'anonymous')",
    )

    # --- games ---
    op.drop_constraint(
        "fk_games_user_id_users",
        "games",
        type_="foreignkey",
    )
    op.alter_column(
        "games",
        "user_id",
        existing_type=sa.Integer(),
        type_=sa.String(length=64),
        nullable=True,
        postgresql_using="user_id::text",
    )
