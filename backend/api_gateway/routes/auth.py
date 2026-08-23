"""Endpoint'ы регистрации и авторизации (задача 64).

Контракт для фронтенда (RegisterModal):
- POST /api/auth/register {login, password, email?, elo?} → токены + пользователь
- POST /api/auth/login {login, password} → токены + пользователь (login или email)
- POST /api/auth/refresh {refresh_token} → новая пара токенов (ротация)
- POST /api/auth/logout {refresh_token} → отзыв refresh-токена
- GET  /api/auth/me (Authorization: Bearer <access>) → данные пользователя

Ошибки — единый формат {"detail": "..."} с кодами 400/401/409/422.
"""

import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_current_user
from backend.api_gateway.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_expiry,
    verify_password,
)
from backend.db.session import get_db
from backend.models.user import RefreshToken, User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    login: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=72)
    email: str | None = Field(default=None, max_length=255)
    elo: int | None = Field(default=None, ge=100, le=3500)

    @field_validator("login")
    @classmethod
    def login_charset(cls, v: str) -> str:
        # \w в Python юникодный: разрешаем и кириллицу, и латиницу
        if not re.fullmatch(r"[\w-]+", v):
            raise ValueError("Логин: только буквы (рус/eng), цифры, _ и -")
        return v


class LoginRequest(BaseModel):
    login: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=72)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=16, max_length=256)


async def _issue_tokens(db: AsyncSession, user: User) -> dict:
    raw_refresh, token_hash = generate_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id, token_hash=token_hash, expires_at=refresh_expiry()
        )
    )
    await db.commit()
    return {
        "access_token": create_access_token(user.id, user.login, user.is_admin),
        "refresh_token": raw_refresh,
        "token_type": "bearer",
        "user": user.public(),
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Создаёт аккаунт. Пароль хешируется bcrypt (cost 12), в БД только хеш."""
    existing = await db.scalar(select(User).where(User.login == req.login))
    if existing:
        raise HTTPException(status_code=409, detail="Логин уже занят")
    if req.email:
        existing_email = await db.scalar(select(User).where(User.email == req.email))
        if existing_email:
            raise HTTPException(status_code=409, detail="Email уже занят")

    user = User(
        login=req.login,
        email=req.email or None,
        elo=req.elo,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    await db.flush()
    tokens = await _issue_tokens(db, user)
    return {"message": "Аккаунт создан", **tokens}


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Вход по логину или e-mail. На несовпадение пары — одинаковый 401."""
    user = await db.scalar(
        select(User).where((User.login == req.login) | (User.email == req.login))
    )
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    return {"message": "Вход выполнен", **(await _issue_tokens(db, user))}


@router.post("/refresh")
async def refresh(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Ротация refresh-токена: старый отзывается, выдаётся новая пара."""
    token_hash = hash_refresh_token(req.refresh_token)
    row = await db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    now = datetime.now(UTC)
    if (
        not row
        or row.revoked
        or row.expires_at.replace(tzinfo=UTC) < now
    ):
        raise HTTPException(status_code=401, detail="Refresh-токен недействителен")

    user = await db.get(User, row.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    row.revoked = True  # ротация
    tokens = await _issue_tokens(db, user)
    return {"message": "Токены обновлены", **tokens}


@router.post("/logout")
async def logout(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Отзывает refresh-токен (access доживёт до конца своего короткого TTL)."""
    token_hash = hash_refresh_token(req.refresh_token)
    row = await db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    if row and not row.revoked:
        row.revoked = True
        await db.commit()
    return {"ok": True}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    """Данные текущего пользователя по access-токену."""
    return {"user": user.public()}


@router.get("/admin-only")
async def admin_only(user: User = Depends(get_current_user)):
    """Пример защищённого ресурса: доступ только для администраторов."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Нужны права администратора")
    return {"ok": True, "secret": f"Секретный дамп для {user.login}"}
