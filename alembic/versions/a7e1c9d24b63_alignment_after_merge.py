"""Выравнивание схемы после слияния анкеты и ролей

Revision ID: a7e1c9d24b63
Revises: cfd7967d8c5c
Create Date: 2026-10-07

- users.role: String(16) из e5f4b3a2c1d0 -> String(32) как в модели;
- users.rating_scale: String(32) -> String(64) как в модели;
- users.parental_consent: колонка их assessment-потока, у нас ответ
  хранится внутри users.onboarding — удаляем (модель её не знает);
- ix_training_reviews_user_due: составной индекс из e4a51c7d92f1
  на случай БД, где он не был создан;
- chat_messages.ts: fb22cbd6384b создаёт Float, модель (Python float)
  соответствует Double — приводим колонку к Double.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7e1c9d24b63"
down_revision: str | Sequence[str] | None = "cfd7967d8c5c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _columns(table_name: str) -> dict[str, sa.types.TypeEngine]:
    bind = op.get_bind()
    return {col["name"]: col["type"] for col in sa.inspect(bind).get_columns(table_name)}


def upgrade() -> None:
    users = _columns("users")

    role_type = users.get("role")
    if isinstance(role_type, sa.String) and (role_type.length or 0) < 32:
        with op.batch_alter_table("users") as batch_op:
            batch_op.alter_column(
                "role",
                type_=sa.String(length=32),
                existing_type=sa.String(length=role_type.length or 16),
                existing_nullable=False,
                server_default="learner",
            )

    rating_scale_type = users.get("rating_scale")
    if isinstance(rating_scale_type, sa.String) and (rating_scale_type.length or 0) < 64:
        with op.batch_alter_table("users") as batch_op:
            batch_op.alter_column(
                "rating_scale",
                type_=sa.String(length=64),
                existing_type=sa.String(length=rating_scale_type.length or 32),
                existing_nullable=True,
            )

    if "parental_consent" in users:
        op.drop_column("users", "parental_consent")

    index_names = {
        ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes("training_reviews")
    }
    if "ix_training_reviews_user_due" not in index_names:
        op.create_index(
            "ix_training_reviews_user_due",
            "training_reviews",
            ["user_id", "next_review_at"],
            unique=False,
        )

    ts_type = _columns("chat_messages").get("ts")
    if ts_type is not None and type(ts_type).__name__ != "Double":
        with op.batch_alter_table("chat_messages") as batch_op:
            batch_op.alter_column(
                "ts",
                type_=sa.Double(),
                existing_type=sa.Float(),
                existing_nullable=False,
            )


def downgrade() -> None:
    op.add_column("users", sa.Column("parental_consent", sa.Boolean(), nullable=True))
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "role",
            type_=sa.String(length=16),
            existing_type=sa.String(length=32),
            existing_nullable=False,
            server_default="learner",
        )
        batch_op.alter_column(
            "rating_scale",
            type_=sa.String(length=32),
            existing_type=sa.String(length=64),
            existing_nullable=True,
        )
    op.drop_index("ix_training_reviews_user_due", table_name="training_reviews")
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.alter_column(
            "ts",
            type_=sa.Float(),
            existing_type=sa.Double(),
            existing_nullable=False,
        )
