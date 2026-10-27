"""GigaCoin balance, reward limits, deduplication and active-day streaks."""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.models  # noqa: F401 - register all tables in metadata
from backend.api_gateway.models import FinishGameRequest
from backend.api_gateway.routes.game import _verified_maia_result
from backend.db.base import Base
from backend.models.gigacoin_event import GigacoinEvent
from backend.models.user import User
from backend.services.gigacoin_service import (
    DAILY_CAP,
    MAIA_WIN_COINS,
    PUZZLE_DAILY_CAP,
    award_gigacoins,
    maia_win_reward,
    puzzle_reward,
)


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        user = User(
            login="test-user",
            email=None,
            password_hash="test",
        )
        session.add(user)
        await session.flush()
        yield session, user
    await engine.dispose()


def test_reward_tables_match_the_proposed_difficulty_curve():
    assert [puzzle_reward(rating) for rating in (700, 1199, 1200, 1799, 1800, 2400)] == [1, 1, 2, 2, 3, 3]
    assert list(MAIA_WIN_COINS.values()) == [2, 3, 4, 5, 7, 10, 14]
    assert maia_win_reward(1100) == 2
    assert maia_win_reward(2600) == 14
    assert maia_win_reward(3600) == 0  # Stockfish isn't one of Maia's levels.


def test_game_reward_requires_a_replayed_terminal_result():
    # Fool's Mate is real, but too short to count as a rated/rewarded game.
    request = FinishGameRequest(
        moves=["f2f3", "e7e5", "g2g4", "d8h4"],
        elo=1100,
        engine="maia3",
        result="0-1",
        status="checkmate",
        player_color="b",
    )
    assert _verified_maia_result(request) is None


@pytest.mark.asyncio
async def test_award_is_idempotent_and_puzzle_rewards_have_their_own_cap(db):
    session, user = db
    now = datetime(2026, 10, 21, 9, tzinfo=timezone.utc)

    assert await award_gigacoins(session, user, "puzzle", "puzzle-1", 3, now=now) == 3
    assert await award_gigacoins(session, user, "puzzle", "puzzle-1", 3, now=now) == 0
    puzzle_total = 3
    for index in range(2, 20):
        puzzle_total += await award_gigacoins(
            session, user, "puzzle", f"puzzle-{index}", 3, now=now
        )
    assert puzzle_total == PUZZLE_DAILY_CAP

    # All ordinary activities share the 50-coin cap; puzzles are included in it.
    assert await award_gigacoins(session, user, "maia_game", "game-1", 30, now=now) == DAILY_CAP - PUZZLE_DAILY_CAP
    assert await award_gigacoins(session, user, "maia_game", "game-2", 3, now=now) == 0
    assert user.gigacoins == DAILY_CAP
    event_count = await session.scalar(select(func.count(GigacoinEvent.id)))
    assert event_count == 21


@pytest.mark.asyncio
async def test_each_seventh_active_day_adds_a_streak_bonus(db):
    session, user = db
    first_day = datetime(2026, 10, 15, 9, tzinfo=timezone.utc)

    rewards = [
        await award_gigacoins(
            session,
            user,
            "training_first",
            f"task-{day}",
            2,
            now=first_day + timedelta(days=day),
        )
        for day in range(7)
    ]

    assert rewards[:6] == [2] * 6
    assert rewards[6] == 12  # 2 for the activity and 10 for the seven-day streak.
    assert user.gigacoins == 24
