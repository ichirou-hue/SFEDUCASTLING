"""Общие зависимости FastAPI для авторизации и ролей."""

from collections.abc import Callable

import jwt as pyjwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.security import decode_access_token
from backend.db.session import get_db
from backend.models.user import ROLE_ADMIN, User

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Возвращает текущего пользователя по Bearer access-токену."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    try:
        payload = decode_access_token(credentials.credentials)
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Токен недействителен или истёк")

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Некорректный токен")

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Как get_current_user, но для анонимного запроса возвращает None."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (pyjwt.PyJWTError, KeyError, TypeError, ValueError):
        return None
    return await db.get(User, user_id)


def require_roles(*allowed_roles: str) -> Callable:
    """Фабрика FastAPI dependency для RBAC.

    Пример:
        user: User = Depends(require_roles("admin"))
    """
    allowed = set(allowed_roles)

    async def _dependency(user: User = Depends(get_current_user)) -> User:
        if user.effective_role not in allowed:
            raise HTTPException(status_code=403, detail="Недостаточно прав")
        return user

    return _dependency


async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    if user.effective_role != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Нужны права администратора")
    return user
