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

import math
import re
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_current_admin, get_current_user
from backend.api_gateway.rate_limit import limiter
from backend.api_gateway.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    refresh_expiry,
    verify_password,
)
from backend.config.settings import settings
from backend.db.session import get_db
from backend.models.gigacoin_event import GigacoinEvent
from backend.models.user import ROLE_ADMIN, ROLE_LEARNER, RefreshToken, User
from backend.models.user_chess_account import (
    UserChessAccount,
    VERIFICATION_VERIFIED,
)
from backend.services.assessment import onboarding_rating
from backend.services.login_throttle import LoginThrottle
from backend.services.nickname import nickname_change_price, validate_nickname
from backend.services.user_activity import record_user_visit
from backend.services.gigacoin_service import ONBOARDING_REWARD, award_gigacoins

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Защита входа. Ключ «IP + логин» не даёт заблокировать чужой аккаунт с
# другого адреса (обычная проблема глобального локаута по логину), а
# отдельный ключ на IP мешает перебирать разные логины с одного места.
_login_throttle = LoginThrottle(
    after_failures=settings.auth.login_backoff_after,
    base_delay=settings.auth.login_backoff_base_sec,
    max_delay=settings.auth.login_backoff_max_sec,
    ttl=settings.auth.login_failure_ttl_sec,
)
_ip_throttle = LoginThrottle(
    after_failures=settings.auth.login_ip_backoff_after,
    base_delay=settings.auth.login_backoff_base_sec,
    max_delay=settings.auth.login_backoff_max_sec,
    ttl=settings.auth.login_failure_ttl_sec,
)


def _login_key(raw: str) -> str:
    """Ключ попыток: логин/email без регистра и лишних пробелов."""
    return f"login:{(raw or '').strip().lower()}"


def _ip_key(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def _too_many_attempts(retry_after: float) -> HTTPException:
    """429 с Retry-After: клиент ждёт сам, сервер не блокирует воркер."""
    seconds = max(1, math.ceil(retry_after))
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            "Слишком много неудачных попыток входа. "
            f"Повторите через {seconds} с."
        ),
        headers={"Retry-After": str(seconds)},
    )


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


class NicknameChangeRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=64)


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
    # login становится стартовым никнеймом, поэтому проверяем занятость
    # и по логинам, и по отображаемым именам (без учёта регистра).
    existing = await db.scalar(
        select(User).where(
            or_(
                User.login == req.login,
                func.lower(User.nickname) == req.login.lower(),
            )
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail="Логин уже занят")
    if req.email:
        existing_email = await db.scalar(select(User).where(User.email == req.email))
        if existing_email:
            raise HTTPException(status_code=409, detail="Email уже занят")

    user = User(
        login=req.login,
        nickname=req.login,
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
@limiter.limit(settings.auth.login_rate_limit)
async def login(
    request: Request,
    req: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    login_key = _login_key(req.login)
    ip_key = _ip_key(request)

    # Уже в ожидании — сразу 429, не тратя bcrypt и запрос к БД.
    blocked_for = max(
        _login_throttle.retry_after(login_key),
        _ip_throttle.retry_after(ip_key),
    )
    if blocked_for > 0:
        raise _too_many_attempts(blocked_for)

    user = await db.scalar(
        select(User).where((User.login == req.login) | (User.email == req.login))
    )
    if not user or not verify_password(req.password, user.password_hash):
        delay = max(
            _login_throttle.record_failure(login_key),
            _ip_throttle.record_failure(ip_key),
        )
        if delay > 0:
            raise _too_many_attempts(delay)
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    # Успех снимает блокировку логина. IP-счётчик не сбрасываем: иначе
    # перебор можно было бы «обнулять» входом в свой же аккаунт.
    _login_throttle.record_success(login_key)
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


@router.patch("/nickname")
async def change_nickname(
    req: NicknameChangeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Меняет отображаемый никнейм за GigaCoin.

    Логин не затрагивается: это идентификатор входа. Цена удваивается
    после каждой смены (10, 20, 40, ...). Баланс, счётчик смен и запись
    в журнале GigaCoin обновляются в одной транзакции под блокировкой
    строки пользователя.
    """
    try:
        new_nickname = validate_nickname(req.nickname)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    locked = await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if locked is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    current_nickname = locked.nickname or locked.login
    if new_nickname.casefold() == current_nickname.casefold():
        raise HTTPException(status_code=400, detail="Новый никнейм совпадает с текущим")

    # Никнейм не должен повторять ни чужой никнейм, ни чужой логин:
    # логин остаётся идентификатором входа и не должен выглядеть как
    # чужое отображаемое имя.
    conflict = await db.scalar(
        select(User.id)
        .where(
            User.id != locked.id,
            or_(
                func.lower(User.nickname) == new_nickname.lower(),
                func.lower(User.login) == new_nickname.lower(),
            ),
        )
        .limit(1)
    )
    if conflict is not None:
        raise HTTPException(status_code=409, detail="Этот никнейм уже занят")

    changes = int(locked.nickname_changes or 0)
    price = nickname_change_price(changes)
    balance = int(locked.gigacoins or 0)
    if balance < price:
        raise HTTPException(
            status_code=409,
            detail=f"Недостаточно GigaCoin: смена стоит {price}",
        )

    now = datetime.now(UTC)
    locked.gigacoins = balance - price
    locked.nickname = new_nickname
    locked.nickname_changes = changes + 1
    locked.nickname_changed_at = now
    db.add(
        GigacoinEvent(
            user_id=locked.id,
            reason="nickname_change",
            ref=f"{changes + 1}:{new_nickname}"[:128],
            amount=-price,
            created_at=now,
        )
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        # Параллельная смена на тот же никнейм: уникальный индекс БД
        # остаётся последней линией защиты.
        await db.rollback()
        raise HTTPException(status_code=409, detail="Этот никнейм уже занят") from exc
    await db.refresh(locked)
    return {
        "ok": True,
        "user": locked.public(),
        "price_paid": price,
        "next_price": nickname_change_price(changes + 1),
    }


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
    gigacoins_earned = await award_gigacoins(
        db, user, "onboarding", "initial-questionnaire", ONBOARDING_REWARD
    )
    await db.commit()
    return {
        "message": "Анкета сохранена",
        "user": user.public(),
        "gigacoins_earned": gigacoins_earned,
        "gigacoins_balance": int(user.gigacoins or 0),
    }


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
