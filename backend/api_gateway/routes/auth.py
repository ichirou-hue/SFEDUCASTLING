"""Регистрация, авторизация, долговременные сессии и роли пользователей."""

import re
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_current_admin, get_current_user
from backend.api_gateway.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_expiry,
    verify_password,
)
from backend.db.session import get_db
from backend.models.user import ROLE_ADMIN, ROLE_LEARNER, RefreshToken, User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    login: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=72)
    email: str | None = Field(default=None, max_length=255)
    elo: int | None = Field(default=None, ge=100, le=3500)

    @field_validator("login")
    @classmethod
    def login_charset(cls, v: str) -> str:
        if not re.fullmatch(r"[\w-]+", v):
            raise ValueError("Логин: только буквы (рус/eng), цифры, _ и -")
        return v


class LoginRequest(BaseModel):
    login: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=72)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=16, max_length=256)


class RoleUpdateRequest(BaseModel):
    role: Literal["learner", "admin"]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def _issue_tokens(db: AsyncSession, user: User) -> dict:
    raw_refresh, token_hash = generate_refresh_token()
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=refresh_expiry(),
        )
    )
    await db.commit()
    return {
        "access_token": create_access_token(
            user.id,
            user.login,
            user.effective_role,
        ),
        "refresh_token": raw_refresh,
        "token_type": "bearer",
        "user": user.public(),
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
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
        role=ROLE_LEARNER,
    )
    db.add(user)
    await db.flush()
    tokens = await _issue_tokens(db, user)
    return {"message": "Аккаунт создан", **tokens}


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(
        select(User).where((User.login == req.login) | (User.email == req.login))
    )
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    return {"message": "Вход выполнен", **(await _issue_tokens(db, user))}


@router.post("/refresh")
async def refresh(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Ротация refresh-токена и продление активной сессии ещё на 90 дней."""
    token_hash = hash_refresh_token(req.refresh_token)
    row = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    now = datetime.now(UTC)
    if not row or row.revoked or _as_utc(row.expires_at) < now:
        raise HTTPException(status_code=401, detail="Refresh-токен недействителен")

    user = await db.get(User, row.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    # Старый refresh больше использовать нельзя; новая пара получает новый TTL.
    row.revoked = True
    tokens = await _issue_tokens(db, user)
    return {"message": "Токены обновлены", **tokens}


@router.post("/logout")
async def logout(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    token_hash = hash_refresh_token(req.refresh_token)
    row = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if row and not row.revoked:
        row.revoked = True
        await db.commit()
    return {"ok": True}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"user": user.public()}


@router.get("/admin-only")
async def admin_only(user: User = Depends(get_current_admin)):
    return {
        "ok": True,
        "role": user.effective_role,
        "secret": f"Секретный дамп для {user.login}",
    }


@router.get("/admin/users")
async def admin_users(
    limit: int = Query(default=100, ge=1, le=500),
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Список пользователей для будущей админ-панели."""
    rows = await db.scalars(select(User).order_by(User.id).limit(limit))
    users = [user.public() for user in rows.all()]
    return {"total": len(users), "items": users}


@router.patch("/admin/users/{user_id}/role")
async def admin_change_user_role(
    user_id: int,
    req: RoleUpdateRequest,
    admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Меняет роль пользователя. Администратор не может снять роль сам у себя."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    if user.id == admin.id and req.role != ROLE_ADMIN:
        raise HTTPException(
            status_code=400,
            detail="Нельзя снять роль администратора у самого себя",
        )

    user.set_role(req.role)
    await db.commit()
    await db.refresh(user)
    return {"ok": True, "user": user.public()}
