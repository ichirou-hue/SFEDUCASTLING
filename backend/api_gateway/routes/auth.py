"""Endpoint'ы регистрации, авторизации, сессий и ролей.

Контракт для фронтенда (RegisterModal, OnboardingModal):
- POST /api/auth/register {login, password, email?, elo?} → токены + пользователь
- POST /api/auth/login {login, password} → токены + пользователь (login или email)
- POST /api/auth/refresh {refresh_token} → новая пара токенов (ротация)
- POST /api/auth/logout {refresh_token} → отзыв refresh-токена
- GET  /api/auth/me (Authorization: Bearer <access>) → данные пользователя
- POST /api/auth/onboarding {answers} → сохранение анкеты, user.onboarding
- GET  /api/auth/admin/users, PATCH .../role → управление ролями (admin)

Ошибки — единый формат {"detail": "..."} с кодами 400/401/409/422.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator
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
from backend.models.user_chess_account import (
    UserChessAccount,
    VERIFICATION_VERIFIED,
)
from backend.services.assessment import onboarding_rating
from backend.services.user_activity import record_user_visit

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    login: str
    password: str
    email: str | None = Field(default=None, max_length=255)
    elo: int | None = Field(default=None, ge=100, le=3500)

    @field_validator("login")
    @classmethod
    def login_valid(cls, v: str) -> str:
        # Обычный для авторизации безопасный набор: латиница, цифры и
        # спецсимволы _ - . ! @ # $ % ^ & * + = ~ . Пробелы, кавычки (", '),
        # национальная пунктуация и кириллица не принимаются.
        if len(v) < 3:
            raise ValueError("Логин: минимум 3 символа")
        if len(v) > 32:
            raise ValueError("Логин: максимум 32 символа")
        if not re.fullmatch(r"[A-Za-z0-9_.\-!@#$%^&*+=~]+", v):
            raise ValueError("Логин: только латинские буквы, цифры и спецсимволы")
        return v

    @field_validator("password")
    @classmethod
    def password_valid(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Пароль: минимум 8 символов")
        if len(v) > 72:
            raise ValueError("Пароль: максимум 72 символа")
        if not re.fullmatch(r"[\x21-\x7E]+", v):
            raise ValueError("Пароль: только латинские буквы, цифры и спецсимволы")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Пароль: нужна хотя бы одна латинская буква")
        if not re.search(r"\d", v):
            raise ValueError("Пароль: нужна хотя бы одна цифра")
        return v


class LoginRequest(BaseModel):
    login: str
    password: str

    @field_validator("login")
    @classmethod
    def login_length(cls, v: str) -> str:
        if not 3 <= len(v) <= 255:
            raise ValueError("Логин: от 3 до 255 символов")
        return v

    @field_validator("password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if not 1 <= len(v) <= 72:
            raise ValueError("Пароль: от 1 до 72 символов")
        return v


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=16, max_length=256)


# === Онбординг-анкета Q1-Q8 (раздел 1 методики) =============================
#
# Ответы хранятся кодами (stables), а не текстами — тексты меняются,
# коды нет. Сырые ответы лежат в users.onboarding; признак заполнения —
# users.onboarding IS NOT NULL.

Q1_CHOICES = {"never", "know_moves", "sometimes", "regular", "tournaments"}
Q3_CHOICES = {"family", "online_rating", "tournaments", "child"}
Q4_CHOICES = {"lt1", "h1_3", "h3_5", "h5plus"}
Q5_CHOICES = {"puzzles", "games", "lessons"}
Q6_CHOICES = {"always", "stuck", "never"}
Q7_CHOICES = {"u10", "age10_16", "age17plus"}
Q8_CHOICES = {"coach", "friends", "internet", "school"}

# prior_band и «сырая» оценка теперь считаются в services.assessment
# (onboarding_rating): там же учтена привязка внешнего рейтинга и ограничение
# самооценки без привязки.

# TODO(BACKEND): методика, раздел 1 — сверка и доработка по мере развития.
#
# 1) Q2 сейчас принимается на веру: рейтинг присылает фронт. По методике его
#    нужно проверять через GET /api/chess-profile (Lichess / Chess.com):
#      - значимый рейтинг: >= 30 партий, RD < 110, активность за год;
#      - не прошёл проверку -> откат к Q1 (как и сейчас, но осознанно);
#      - приоритет источников: lichess_blitz -> lichess_rapid -> chesscom_blitz
#        -> chesscom_rapid -> bullet (с пометкой «пуля шумная»).
#    Результат писать в users.rating_estimate (шкала Lichess blitz) и
#    users.rating_scale (напр. "lichess_blitz"), а не только в onboarding JSON.
#
# 2) Отдельные колонки вместо вложенности в users.onboarding:
#      users.prior_band   int 1..4   — считается здесь, но должен жить в колонке;
#      users.rating_estimate int     — шкала Lichess blitz;
#      users.rating_scale str        — происхождение шкалы;
#      users.pedagogy     str        — ответ Q6 (подсказки);
#      users.skill_band   int 0..4   — пишет входной тест, не анкета.
#    Пока эти колонок нет — источник истины users.onboarding (см. public()).
#
# 3) parental_consent: сейчас только валидация ответа в JSON. Нужно:
#      - отдельный флаг + контакт для PII-логики (раздел 8 методики);
#      - детский режим (упрощённый UI, короткие сессии, минимум логов)
#        и гостевой режим без согласия;
#      - 10-16: уведомление родителя рекомендовано (уточнить по юрисдикции).
#
# 4) prior_band используется ТОЛЬКО для флага расхождения с результатом
#    входного теста (|assess - prior| >= 3 -> «перепройти», кулдаун 7 дней).
#    Для отбора задач он НЕ применяется — тест фиксированный.


class Q2Answer(BaseModel):
    platform: str = Field(pattern="^(lichess|chesscom)$")
    login: str = Field(min_length=1, max_length=64)
    time_control: str = Field(pattern="^(blitz|rapid|bullet)$")
    rating: int | None = Field(default=None, ge=0, le=4000)


class ParentalConsent(BaseModel):
    granted: bool
    contact: str | None = Field(default=None, max_length=255)


class OnboardingRequest(BaseModel):
    q1: str
    q2: Q2Answer | None = None
    q3: list[str] = Field(min_length=1, max_length=2)
    q4: str
    q5: list[str] = Field(min_length=3, max_length=3)
    q6: str
    q7: str
    q8: str | None = None
    parental_consent: ParentalConsent | None = None

    @field_validator("q1")
    @classmethod
    def _q1(cls, v: str) -> str:
        if v not in Q1_CHOICES:
            raise ValueError(f"q1: допустимы {sorted(Q1_CHOICES)}")
        return v

    @field_validator("q3")
    @classmethod
    def _q3(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("q3: ответы не должны повторяться")
        bad = set(v) - Q3_CHOICES
        if bad:
            raise ValueError(f"q3: неизвестные ответы {sorted(bad)}")
        return v

    @field_validator("q4")
    @classmethod
    def _q4(cls, v: str) -> str:
        if v not in Q4_CHOICES:
            raise ValueError(f"q4: допустимы {sorted(Q4_CHOICES)}")
        return v

    @field_validator("q5")
    @classmethod
    def _q5(cls, v: list[str]) -> list[str]:
        # «ранжировать» — все три варианта, каждый ровно один раз
        if sorted(v) != sorted(Q5_CHOICES):
            raise ValueError(f"q5: нужен полный порядок из {sorted(Q5_CHOICES)}")
        return v

    @field_validator("q6")
    @classmethod
    def _q6(cls, v: str) -> str:
        if v not in Q6_CHOICES:
            raise ValueError(f"q6: допустимы {sorted(Q6_CHOICES)}")
        return v

    @field_validator("q7")
    @classmethod
    def _q7(cls, v: str) -> str:
        if v not in Q7_CHOICES:
            raise ValueError(f"q7: допустимы {sorted(Q7_CHOICES)}")
        return v

    @field_validator("q8")
    @classmethod
    def _q8(cls, v: str | None) -> str | None:
        if v is not None and v not in Q8_CHOICES:
            raise ValueError(f"q8: допустимы {sorted(Q8_CHOICES)}")
        return v

    @model_validator(mode="after")
    def _check_parental_consent(self) -> OnboardingRequest:
        # Младше 13 (группа «до 10») — обязательно согласие родителя/опекуна.
        if self.q7 == "u10":
            if not self.parental_consent:
                raise ValueError("parental_consent обязателен для возрастной группы «до 10»")
            if not self.parental_consent.granted:
                raise ValueError("Без согласия родителя анкету принять нельзя")
        return self


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
            token_version=user.token_version,
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
    # День регистрации сразу считается первым днём серии.
    await record_user_visit(db, user.id)
    tokens = await _issue_tokens(db, user)
    return {"message": "Аккаунт создан", **tokens}


@router.post("/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(
        select(User).where((User.login == req.login) | (User.email == req.login))
    )
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    await record_user_visit(db, user.id)
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

    # Возврат пользователя с валидным refresh-токеном — тоже визит.
    await record_user_visit(db, user.id)

    # Старый refresh больше использовать нельзя; новая пара получает новый TTL.
    row.revoked = True
    tokens = await _issue_tokens(db, user)
    return {"message": "Токены обновлены", **tokens}


@router.post("/logout")
async def logout(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Отменяет refresh-токен и мгновенно отзывает все access-токены пользователя.

    Access-токены stateless, поэтому отзываем их версией сессии: инкремент
    users.token_version делает недействительными все ранее выданные JWT.
    """
    token_hash = hash_refresh_token(req.refresh_token)
    row = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if row and not row.revoked:
        row.revoked = True
        user = await db.get(User, row.user_id)
        if user:
            user.token_version += 1
        await db.commit()
    return {"ok": True}


@router.get("/me")
async def me(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Открытие приложения (фронт вызывает /me на старте) отмечает визит,
    # даже если access-токен ещё жив и refresh не понадобился.
    await record_user_visit(db, user.id)
    return {"user": user.public()}


async def _linked_rating(db: AsyncSession, user_id: int) -> dict[str, Any] | None:
    """Последний подтверждённый внешний рейтинг для стартовой оценки.

    Возвращает ``{"rating", "scale", "usable"}`` или None. Привязка — единственный
    способ получить стартовую оценку выше ограничения самооценки.
    """
    account = await db.scalar(
        select(UserChessAccount)
        .where(
            UserChessAccount.user_id == user_id,
            UserChessAccount.verified.is_(True),
            UserChessAccount.verification_status == VERIFICATION_VERIFIED,
            UserChessAccount.rating_usable.is_(True),
        )
        .order_by(UserChessAccount.updated_at.desc(), UserChessAccount.id.desc())
    )
    if account is None or account.rating is None:
        return None
    return {
        "rating": int(account.rating),
        "scale": account.rating_scale,
        "usable": True,
        "platform": account.platform,
        "username": account.username,
        "rating_type": account.rating_type,
        "games": int(account.games or 0),
    }


@router.post("/onboarding")
async def save_onboarding(
    req: OnboardingRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Сохраняет ответы онбординг-анкеты Q1-Q8.

    Повторный вызов перезаписывает ответы (анкету можно пройти заново).
    После успеха user.onboarding != null — фронтенд прячет плашку
    «Заполните анкету».

    TODO(BACKEND): контракт с фронтендом стабилен и менять его не нужно —
    фронт отправляет коды ответов (q1..q8), а не тексты. Доработать внутри:
    разложить ответы по колонкам (prior_band / rating_estimate / rating_scale /
    pedagogy), проверить Q2 через /api/chess-profile, вынести parental consent
    в отдельный флаг (см. блок TODO в начале файла).
    """
    payload = req.model_dump(mode="json")
    # Стартовая оценка: внешний привязанный рейтинг (если есть) или самооценка
    # Q1 с ограничением SELF_REPORT_CAP. Пишем и в JSON, и в отдельные колонки —
    # подбор пазлов и стартовая сложность опираются на колонки.
    linked = await _linked_rating(db, user.id)
    rating_estimate, rating_scale, prior_band = onboarding_rating(payload, linked=linked)
    user.onboarding = {
        "answers": payload,
        "prior_band": prior_band,
        "linked_rating": linked,
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    user.prior_band = prior_band
    user.rating_estimate = rating_estimate
    user.rating_scale = rating_scale
    await db.commit()
    return {"message": "Анкета сохранена", "user": user.public()}


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
