"""Серверные награды GigaChess: дневные лимиты, баланс и серии активности."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.gigacoin_event import GigacoinEvent
from backend.models.user import User

MOSCOW_TZ = timezone(timedelta(hours=3), name="MSK")
DAILY_CAP = 50
PUZZLE_DAILY_CAP = 30
PUZZLE_COINS = ((1200, 1), (1800, 2), (10_001, 3))
TRAINING_FIRST_CORRECT = 3
TRAINING_REPEAT = 1
ONBOARDING_REWARD = 5
LEVEL_TEST_REWARD = 10
MAIA_WIN_COINS = {1100: 2, 1300: 3, 1500: 4, 1700: 5, 1900: 7, 2200: 10, 2600: 14}
MAIA_DRAW_COINS = 1
MAIA_FAIR_GAP = 100
ACTIVE_STREAK_BONUS = 10
STREAK_CAP_EXEMPT = True


def puzzle_reward(rating: int | None) -> int:
    value = max(0, int(rating or 0))
    if value < PUZZLE_COINS[0][0]:
        return 1
    if value < PUZZLE_COINS[1][0]:
        return 2
    return 3


def maia_win_reward(rating: int | None) -> int:
    """Reward for a configured Maia level; Stockfish's 3600 is excluded."""
    return MAIA_WIN_COINS.get(int(rating or 0), 0)


def moscow_today(now: datetime | None = None) -> date:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(MOSCOW_TZ).date()


def _utc_bounds(now: datetime) -> tuple[datetime, datetime, date]:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    today = now.astimezone(MOSCOW_TZ).date()
    start_local = datetime.combine(today, time.min, tzinfo=MOSCOW_TZ)
    start = start_local.astimezone(timezone.utc)
    return start, start + timedelta(days=1), today


async def _earned_between(db: AsyncSession, user_id: int, start: datetime, end: datetime) -> int:
    return int(await db.scalar(
        select(func.coalesce(func.sum(GigacoinEvent.amount), 0)).where(
            GigacoinEvent.user_id == user_id,
            GigacoinEvent.created_at >= start,
            GigacoinEvent.created_at < end,
            GigacoinEvent.amount > 0,
            GigacoinEvent.reason != "active_streak",
        )
    ) or 0)


async def _active_streak(db: AsyncSession, user_id: int, today: date) -> int:
    start = datetime.combine(today - timedelta(days=370), time.min, tzinfo=MOSCOW_TZ).astimezone(timezone.utc)
    end = datetime.combine(today, time.min, tzinfo=MOSCOW_TZ).astimezone(timezone.utc)
    timestamps = (await db.scalars(
        select(GigacoinEvent.created_at).where(
            GigacoinEvent.user_id == user_id,
            GigacoinEvent.created_at >= start,
            GigacoinEvent.created_at < end,
            GigacoinEvent.amount > 0,
        )
    )).all()
    active_days = {value.replace(tzinfo=timezone.utc).astimezone(MOSCOW_TZ).date() for value in timestamps}
    streak = 0
    cursor = today - timedelta(days=1)
    while cursor in active_days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


async def award_gigacoins(
    db: AsyncSession,
    user: User,
    reason: str,
    ref: str,
    amount: int,
    *,
    now: datetime | None = None,
    exempt_daily_cap: bool = False,
    check_streak: bool = True,
) -> int:
    """Award once per (user, reason, ref); returns the actual credited amount."""
    now = now or datetime.now(timezone.utc)
    existing = await db.scalar(select(GigacoinEvent.id).where(
        GigacoinEvent.user_id == user.id,
        GigacoinEvent.reason == reason,
        GigacoinEvent.ref == ref,
    ).limit(1))
    if existing is not None:
        return 0

    start, end, today = _utc_bounds(now)
    requested = max(0, int(amount))
    credited = requested
    if requested and not exempt_daily_cap:
        credited = min(requested, max(0, DAILY_CAP - await _earned_between(db, user.id, start, end)))
        if reason == "puzzle":
            puzzle_earned = int(await db.scalar(select(func.coalesce(func.sum(GigacoinEvent.amount), 0)).where(
                GigacoinEvent.user_id == user.id,
                GigacoinEvent.reason == "puzzle",
                GigacoinEvent.created_at >= start,
                GigacoinEvent.created_at < end,
                GigacoinEvent.amount > 0,
            )) or 0)
            credited = min(credited, max(0, PUZZLE_DAILY_CAP - puzzle_earned))

    db.add(GigacoinEvent(user_id=user.id, reason=reason, ref=str(ref)[:128], amount=credited, created_at=now))
    user.gigacoins = max(0, int(user.gigacoins or 0)) + credited
    await db.flush()

    # Bonus за каждые 7 подряд активных дней. Считаем реальные начисления,
    # поэтому нулевой capped event не продолжает серию.
    streak_bonus = 0
    if credited and check_streak:
        streak = await _active_streak(db, user.id, today)
        if streak and streak % 7 == 6:
            streak_bonus = await award_gigacoins(
                db, user, "active_streak", today.isoformat(), ACTIVE_STREAK_BONUS,
                now=now, exempt_daily_cap=STREAK_CAP_EXEMPT, check_streak=False,
            )
    return credited + streak_bonus
