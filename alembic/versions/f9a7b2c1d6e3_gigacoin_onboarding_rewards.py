"""Backfill one-time rewards for completed onboarding and level tests.

Revision ID: f9a7b2c1d6e3
Revises: e7f4a1c9b263
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f9a7b2c1d6e3"
down_revision: str | None = "e7f4a1c9b263"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ONBOARDING_REF = "initial-questionnaire"
LEVEL_TEST_REF = "initial-assessment"
ONBOARDING_REWARD = 5
LEVEL_TEST_REWARD = 10


def _add_reward(connection, events, users, user_id: int, reason: str, ref: str, amount: int) -> None:
    exists = connection.execute(
        sa.select(events.c.id).where(
            events.c.user_id == user_id,
            events.c.reason == reason,
            events.c.ref == ref,
        )
    ).first()
    if exists:
        return
    connection.execute(sa.insert(events).values(
        user_id=user_id,
        reason=reason,
        ref=ref,
        amount=amount,
    ))
    connection.execute(
        sa.update(users)
        .where(users.c.id == user_id)
        .values(gigacoins=users.c.gigacoins + amount)
    )


def upgrade() -> None:
    connection = op.get_bind()
    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("onboarding", sa.JSON),
        sa.column("assessment_completed_at", sa.DateTime(timezone=True)),
        sa.column("gigacoins", sa.Integer),
    )
    events = sa.table(
        "gigacoin_events",
        sa.column("id", sa.BigInteger),
        sa.column("user_id", sa.Integer),
        sa.column("reason", sa.String(32)),
        sa.column("ref", sa.String(128)),
        sa.column("amount", sa.Integer),
    )

    for row in connection.execute(
        sa.select(users.c.id, users.c.onboarding, users.c.assessment_completed_at)
    ).mappings():
        onboarding = row["onboarding"]
        if isinstance(onboarding, dict) and onboarding.get("submitted_at"):
            _add_reward(
                connection, events, users, row["id"],
                "onboarding", ONBOARDING_REF, ONBOARDING_REWARD,
            )
        if row["assessment_completed_at"] is not None:
            _add_reward(
                connection, events, users, row["id"],
                "level_test", LEVEL_TEST_REF, LEVEL_TEST_REWARD,
            )


def downgrade() -> None:
    connection = op.get_bind()
    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("gigacoins", sa.Integer),
    )
    events = sa.table(
        "gigacoin_events",
        sa.column("user_id", sa.Integer),
        sa.column("reason", sa.String(32)),
        sa.column("ref", sa.String(128)),
        sa.column("amount", sa.Integer),
    )
    for row in connection.execute(
        sa.select(events.c.user_id, events.c.amount).where(
            sa.or_(
                sa.and_(events.c.reason == "onboarding", events.c.ref == ONBOARDING_REF),
                sa.and_(events.c.reason == "level_test", events.c.ref == LEVEL_TEST_REF),
            )
        )
    ).mappings():
        connection.execute(
            sa.update(users)
            .where(users.c.id == row["user_id"])
            .values(gigacoins=sa.case(
                (users.c.gigacoins >= row["amount"], users.c.gigacoins - row["amount"]),
                else_=0,
            ))
        )
    connection.execute(
        sa.delete(events).where(
            sa.or_(
                sa.and_(events.c.reason == "onboarding", events.c.ref == ONBOARDING_REF),
                sa.and_(events.c.reason == "level_test", events.c.ref == LEVEL_TEST_REF),
            )
        )
    )
