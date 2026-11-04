"""Достижения за первые партии и награды за них.

Правила намеренно проверяются повтором партии на сервере: клиентские
``result`` и ``status`` сами по себе не считаются доказательством.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import chess
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.user import User
from backend.models.user_achievement import UserAchievement
from backend.services.gigacoin_service import award_gigacoins

FIRST_GAME = "first_game"
FIRST_MATE = "first_mate"
FIRST_LOSS = "first_loss"

# Минимум 4 полухода: две полные пары ходов. Более короткие записи
# считаем не партией, а тестовым фрагментом.
MIN_GAME_PLIES = 4


@dataclass(frozen=True)
class AchievementDefinition:
    code: str
    title: str
    description: str
    reward: int
    icon: str


ACHIEVEMENTS: dict[str, AchievementDefinition] = {
    FIRST_GAME: AchievementDefinition(
        code=FIRST_GAME,
        title="Первая партия",
        description="Завершите первую партию с ботом",
        reward=10,
        icon="♟",
    ),
    FIRST_MATE: AchievementDefinition(
        code=FIRST_MATE,
        title="Первый мат",
        description="Поставьте мат боту",
        reward=25,
        icon="♛",
    ),
    FIRST_LOSS: AchievementDefinition(
        code=FIRST_LOSS,
        title="Первое поражение",
        description="Проиграйте первую партию: это тоже опыт",
        reward=5,
        icon="♞",
    ),
}


@dataclass(frozen=True)
class VerifiedGame:
    status: str
    result: str


def replay_terminal_game(moves: list[str]) -> VerifiedGame | None:
    """Повторяет партию и возвращает её настоящий терминальный исход.

    Возвращает ``None``, если ходы нелегальны, партия не завершена или
    результат невозможно определить.
    """
    board = chess.Board()
    for token in moves:
        if board.is_game_over(claim_draw=True):
            return None
        try:
            move = chess.Move.from_uci(token)
        except (TypeError, ValueError):
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)

    if not board.is_game_over(claim_draw=True):
        return None

    if board.is_checkmate():
        status = "checkmate"
    elif board.is_stalemate():
        status = "stalemate"
    else:
        status = "draw"

    result = board.result(claim_draw=True)
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return None
    return VerifiedGame(status=status, result=result)


def evaluate_game_achievements(
    moves: list[str],
    result: str,
    player_color: str,
) -> list[str]:
    """Возвращает коды достижений, которые заслуживает завершённая партия."""
    if len(moves) < MIN_GAME_PLIES:
        return []

    verified = replay_terminal_game(moves)
    if verified is None or verified.result != result:
        return []

    codes = [FIRST_GAME]
    player_won = verified.result == ("1-0" if player_color == "w" else "0-1")
    player_lost = verified.result == ("0-1" if player_color == "w" else "1-0")

    if player_won and verified.status == "checkmate":
        codes.append(FIRST_MATE)
    if player_lost:
        codes.append(FIRST_LOSS)
    return codes


def public_achievements(rows: Iterable[UserAchievement]) -> list[dict]:
    """Каталог достижений с отметкой о разблокировке для профиля."""
    unlocked = {row.code: row for row in rows}
    payload: list[dict] = []
    for definition in ACHIEVEMENTS.values():
        row = unlocked.get(definition.code)
        unlocked_at = None
        if row is not None:
            created_at = row.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            unlocked_at = created_at.isoformat()
        payload.append(
            {
                "code": definition.code,
                "title": definition.title,
                "description": definition.description,
                "reward": definition.reward,
                "icon": definition.icon,
                "unlocked": row is not None,
                "awarded": int(row.reward) if row is not None else 0,
                "unlocked_at": unlocked_at,
            }
        )
    return payload


async def unlock_achievements(
    db: AsyncSession,
    user: User,
    codes: Iterable[str],
    *,
    now: datetime | None = None,
) -> list[dict]:
    """Выдаёт достижения и монеты за них ровно один раз.

    Строка пользователя блокируется до проверки существующих достижений,
    чтобы два параллельных запроса не выдали одну награду дважды.
    """
    requested = list(dict.fromkeys(codes))
    if not requested:
        return []

    await db.scalar(select(User.id).where(User.id == user.id).with_for_update())
    existing = set(
        (
            await db.scalars(
                select(UserAchievement.code).where(
                    UserAchievement.user_id == user.id,
                    UserAchievement.code.in_(requested),
                )
            )
        ).all()
    )

    unlocked: list[dict] = []
    for code in requested:
        definition = ACHIEVEMENTS.get(code)
        if definition is None or code in existing:
            continue

        credited = await award_gigacoins(
            db,
            user,
            "achievement",
            code,
            definition.reward,
            now=now,
            exempt_daily_cap=True,
        )
        db.add(
            UserAchievement(
                user_id=user.id,
                code=code,
                reward=credited,
                created_at=now or datetime.now(UTC),
            )
        )
        unlocked.append(
            {
                "code": code,
                "title": definition.title,
                "reward": credited,
                "icon": definition.icon,
            }
        )
        existing.add(code)

    await db.flush()
    return unlocked
