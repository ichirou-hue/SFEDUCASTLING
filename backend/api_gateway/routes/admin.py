"""Админ-эндпоинты: доступ только для роли admin.

Разделение прав доступа:
- guest   — без аккаунта: паззлы, анализ, чат, level-test;
- learner — зарегистрированный пользователь: + учебные модули;
- admin   — роль admin: + администрирование (список пользователей и т.п.).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway import state
from backend.api_gateway.dependecies import require_roles
from backend.db.session import get_db
from backend.models.user import User
from backend.models.user_chess_account import UserChessAccount
from backend.services.adaptive_training import (
    build_weakness_profile,
    compute_learning_progress,
)
from backend.services.training_service import get_training_progress

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Поля, которые админ видит по каждому пользователю (без пароля и токенов).
_USER_FIELDS = (
    "id",
    "login",
    "role",
    "is_admin",
    "elo",
    "skill_band",
    "created_at",
)


@router.get("/users")
async def admin_users(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    """Краткий список пользователей для администратора.

    Гость: 401 (нет токена). Learner: 403 (недостаточно прав).
    Только role=admin получает список.
    """
    rows = await db.scalars(select(User).order_by(User.id))
    public = rows.all()
    user_ids = [u.id for u in public]
    accounts = []
    if user_ids:
        accounts = (
            await db.scalars(
                select(UserChessAccount)
                .where(UserChessAccount.user_id.in_(user_ids))
                .order_by(UserChessAccount.user_id, UserChessAccount.platform)
            )
        ).all()
    ratings_by_user: dict[int, list[dict]] = {}
    for account in accounts:
        ratings_by_user.setdefault(account.user_id, []).append(_public_rating(account))

    return {
        "users": [
            {
                **{k: u.public()[k] for k in _USER_FIELDS},
                "chess_accounts": ratings_by_user.get(u.id, []),
            }
            for u in public
        ]
    }


def _public_rating(account: UserChessAccount) -> dict:
    return {
        "platform": account.platform,
        "username": account.username,
        "rating": account.rating,
        "rating_type": account.rating_type,
        "rating_usable": account.rating_usable,
    }


async def _user_with_ratings(db: AsyncSession, target: User) -> dict:
    accounts = await db.scalars(
        select(UserChessAccount)
        .where(UserChessAccount.user_id == target.id)
        .order_by(UserChessAccount.platform)
    )
    return {
        **target.public(),
        "chess_accounts": [_public_rating(account) for account in accounts],
    }


async def _get_target_or_404(db: AsyncSession, user_id: int) -> User:
    """Проверка существования пользователя; путать «нет доступа» и «нет юзера» нельзя."""
    target = await db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return target


@router.get("/users/{user_id}")
async def admin_user_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_roles("admin")),
):
    """Полный профиль пользователя для администратора (без пароля/токенов)."""
    target = await _get_target_or_404(db, user_id)
    return {"user": await _user_with_ratings(db, target)}


@router.get("/users/{user_id}/stats")
async def admin_user_stats(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_roles("admin")),
):
    """Статистика обучения пользователя для администратора: курс + паззлы + слабые темы."""
    target = await _get_target_or_404(db, user_id)

    weaknesses = None
    if state.puzzle_base:
        weaknesses = await build_weakness_profile(
            db,
            user_id=target.id,
            puzzle_base=state.puzzle_base,
        )

    return {
        "user": await _user_with_ratings(db, target),
        "training": await get_training_progress(db, target.id),
        "learning": await compute_learning_progress(db, user_id=target.id),
        "weaknesses": weaknesses,
    }
