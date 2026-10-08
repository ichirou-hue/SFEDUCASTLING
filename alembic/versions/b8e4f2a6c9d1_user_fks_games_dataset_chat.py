"""games/dataset_moves/chat_messages: user_id -> FK на users.id

Наводим ссылочную целостность (задача «нормальная БД»):
- games.user_id: varchar -> integer FK users.id (ON DELETE SET NULL)
- dataset_moves.user_id: varchar NOT NULL 'anonymous' -> integer NULL FK
- chat_messages: добавлен user_id integer NULL FK

Старые строковые идентификаторы эпохи без авторизации
('test-int', 'user_xxx', 'anonymous', ...) превращаются в NULL:
анонимные данные сохраняются, но без привязки к аккаунту.

Очистка данных выполняется на уровне Python, а не SQL, потому что
CAST(... AS INTEGER) на PostgreSQL падает на нечисловых строках,
а на SQLite молча даёт 0. Смена типа идёт через batch_alter_table:
SQLite не умеет ALTER COLUMN ... TYPE, а на PostgreSQL batch-режим
выполняет обычный ALTER.

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

_CHUNK = 500


def _pg_using(column: str, cast_type: str) -> dict:
    """Приведение varchar->integer нужно только PostgreSQL."""
    if op.get_bind().dialect.name == "postgresql":
        return {"postgresql_using": f"{column}::{cast_type}"}
    return {}


def _valid_user_ids(bind) -> set[int]:
    return {row[0] for row in bind.execute(sa.text("SELECT id FROM users"))}


def _clean_user_ids(bind, table: str) -> None:
    """Нечисловые и отсутствующие в users идентификаторы -> NULL.

    Работает одинаково на обоих диалектах (в отличие от SQL с CAST).
    """
    valid = _valid_user_ids(bind)
    rows = bind.execute(
        sa.text(f"SELECT id, user_id FROM {table} WHERE user_id IS NOT NULL")
    ).fetchall()

    to_null: list[int] = []
    for row_id, raw in rows:
        text = str(raw).strip()
        if text.isdigit() and int(text) in valid:
            continue
        to_null.append(row_id)

    for start in range(0, len(to_null), _CHUNK):
        chunk = to_null[start : start + _CHUNK]
        placeholders = ", ".join(f":i{n}" for n in range(len(chunk)))
        bind.execute(
            sa.text(f"UPDATE {table} SET user_id = NULL WHERE id IN ({placeholders})"),
            {f"i{n}": value for n, value in enumerate(chunk)},
        )


def upgrade() -> None:
    bind = op.get_bind()

    # --- 1. данные: мусорные строковые идентификаторы -> NULL ---
    _clean_user_ids(bind, "games")
    _clean_user_ids(bind, "dataset_moves")

    # --- 2. games.user_id: varchar -> integer FK ---
    with op.batch_alter_table("games") as batch_op:
        batch_op.alter_column(
            "user_id",
            existing_type=sa.String(length=64),
            type_=sa.Integer(),
            nullable=True,
            **_pg_using("user_id", "integer"),
        )
        batch_op.create_foreign_key(
            "fk_games_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # --- 3. dataset_moves.user_id: varchar NOT NULL -> integer NULL FK ---
    # Очистка выполнена выше, пока колонка ещё строковая.
    with op.batch_alter_table("dataset_moves") as batch_op:
        batch_op.alter_column(
            "user_id",
            existing_type=sa.String(length=64),
            type_=sa.Integer(),
            nullable=True,
            server_default=None,
            **_pg_using("user_id", "integer"),
        )
        batch_op.create_foreign_key(
            "fk_dataset_moves_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # --- 4. chat_messages: новая колонка user_id ---
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_chat_messages_user_id", ["user_id"])
        batch_op.create_foreign_key(
            "fk_chat_messages_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    # --- chat_messages: убрать колонку и её следы ---
    op.drop_index("ix_chat_messages_user_id", table_name="chat_messages")
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.drop_constraint(
            "fk_chat_messages_user_id_users", type_="foreignkey"
        )
        batch_op.drop_column("user_id")

    # --- dataset_moves: вернуть varchar NOT NULL 'anonymous' ---
    with op.batch_alter_table("dataset_moves") as batch_op:
        batch_op.drop_constraint(
            "fk_dataset_moves_user_id_users", type_="foreignkey"
        )
        batch_op.alter_column(
            "user_id",
            existing_type=sa.Integer(),
            type_=sa.String(length=64),
            nullable=False,
            server_default=sa.text("'anonymous'"),
            **_pg_using("user_id", "varchar(64)"),
        )

    # --- games: вернуть varchar ---
    with op.batch_alter_table("games") as batch_op:
        batch_op.drop_constraint("fk_games_user_id_users", type_="foreignkey")
        batch_op.alter_column(
            "user_id",
            existing_type=sa.Integer(),
            type_=sa.String(length=64),
            nullable=True,
            **_pg_using("user_id", "varchar(64)"),
        )
