"""Общие зависимости FastAPI.

get_current_user используется любым защищённым роутом:
    user: User = Depends(get_current_user)
"""

import jwt as pyjwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.security import decode_access_token
from backend.db.session import get_db
from backend.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Пользователь по Bearer access-токену (JWT HS256)."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    try:
        payload = decode_access_token(credentials.credentials)
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Токен недействителен или истёк")
    user = await db.get(User, int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user
