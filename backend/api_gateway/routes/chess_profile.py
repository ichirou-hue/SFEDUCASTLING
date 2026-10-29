"""Публичные шахматные профили, soft-link, синхронизация и история верификации.

Soft-link по username позволяет импортировать публичные данные, но НЕ доказывает
владение аккаунтом. Только успешная OAuth-проверка переводит аккаунт в
``verification_status='verified'`` и делает рейтинг доверенным для assessment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.api_gateway.dependecies import get_current_user
from backend.config.settings import settings
from backend.db.session import get_db
from backend.models.user import User
from backend.models.user_chess_account import (
    ChessAccountVerification,
    UserChessAccount,
    VERIFICATION_ERROR,
    VERIFICATION_EXPIRED,
    VERIFICATION_LINKED,
    VERIFICATION_METHOD_USER_ACTION,
    VERIFICATION_PENDING,
    VERIFICATION_REVOKED,
    VERIFICATION_VERIFIED,
)

router = APIRouter(tags=["chess-profile"])

LICHESS_USER_URL = "https://lichess.org/api/user/{username}"
CHESSCOM_PLAYER_URL = "https://api.chess.com/pub/player/{username}"
CHESSCOM_STATS_URL = "https://api.chess.com/pub/player/{username}/stats"

USER_AGENT = settings.chess_oauth.user_agent
HTTP_TIMEOUT = 8.0
RATING_PRIORITY = ("blitz", "rapid", "bullet")
MIN_GAMES_FOR_ASSESSMENT = 30
MAX_LICHESS_RD = 110
MAX_ACTIVITY_AGE_DAYS = 365
# Аккаунт без единой партии не подтверждает уровень: его рейтинг для стартовой
# оценки ограничивается этим порогом (правило онбординга/регистрации).
NO_GAMES_RATING_CAP = 800


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _normalize_platform(value: str) -> str:
    platform = (value or "lichess").lower().strip()
    if platform in ("chess.com", "chesscom", "chess_com", "chess"):
        return "chesscom"
    return "lichess"


def _lichess_profile(username: str):
    try:
        resp = requests.get(
            LICHESS_USER_URL.format(username=username),
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return {"error": "Не удалось связаться с Lichess"}, 502

    if resp.status_code == 404:
        return {"error": "Пользователь не найден на Lichess"}, 404
    if resp.status_code != 200:
        return {"error": f"Lichess ответил {resp.status_code}"}, resp.status_code

    data = resp.json()
    perfs = data.get("perfs", {})
    profile = data.get("profile", {}) or {}

    def _perf(key: str):
        p = perfs.get(key, {}) or {}
        games = int(p.get("games") or 0)
        return {
            "rating": p.get("rating") if games else None,
            "games": games,
            "rd": p.get("rd"),
            "provisional": bool(p.get("prov", False)),
        }

    counts = data.get("count", {}) or {}
    return {
        "platform": "lichess",
        "username": data.get("username", username),
        "title": data.get("title"),
        "name": profile.get("firstName") or None,
        "country": profile.get("flag") or None,
        "avatar": data.get("avatar") or None,
        "last_seen_at": data.get("seenAt"),
        "perfs": {
            "bullet": _perf("bullet"),
            "blitz": _perf("blitz"),
            "rapid": _perf("rapid"),
            "classical": _perf("classical"),
        },
        "counts": {
            "all": int(counts.get("all") or 0),
            "wins": int(counts.get("win") or 0),
            "losses": int(counts.get("loss") or 0),
            "draws": int(counts.get("draw") or 0),
        },
    }, 200


def _chesscom_profile(username: str):
    headers = {"User-Agent": USER_AGENT}
    try:
        player_resp = requests.get(
            CHESSCOM_PLAYER_URL.format(username=username),
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return {"error": "Не удалось связаться с Chess.com"}, 502

    if player_resp.status_code == 404:
        return {"error": "Пользователь не найден на Chess.com"}, 404
    if player_resp.status_code != 200:
        return {"error": f"Chess.com ответил {player_resp.status_code}"}, player_resp.status_code

    data = player_resp.json()
    try:
        stats_resp = requests.get(
            CHESSCOM_STATS_URL.format(username=username),
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        stats = stats_resp.json() if stats_resp.status_code == 200 else {}
    except requests.RequestException:
        stats = {}

    ratings: dict[str, dict[str, Any]] = {}
    counts = {"all": 0, "wins": 0, "losses": 0, "draws": 0}
    for key in ("bullet", "blitz", "rapid", "daily"):
        block = stats.get("chess_" + key, {}) or {}
        last = block.get("last", {}) or {}
        rec = block.get("record", {}) or {}
        games = int(rec.get("win") or 0) + int(rec.get("loss") or 0) + int(rec.get("draw") or 0)
        ratings[key] = {
            # Chess.com отдаёт last.rating даже без партий — скрываем рейтинг,
            # пока не сыграна хотя бы одна партия (как в Lichess-профиле).
            "rating": last.get("rating") if games else None,
            "games": games,
            "rd": None,
            "provisional": False,
        }
        counts["all"] += games
        counts["wins"] += int(rec.get("win") or 0)
        counts["losses"] += int(rec.get("loss") or 0)
        counts["draws"] += int(rec.get("draw") or 0)

    last_online = data.get("last_online")
    return {
        "platform": "chesscom",
        "username": data.get("username", username),
        "title": data.get("title"),
        "name": data.get("name"),
        "country": (data.get("country") or "").rsplit("/", 1)[-1] or None,
        "avatar": data.get("avatar") or None,
        "last_seen_at": int(last_online * 1000) if isinstance(last_online, (int, float)) else None,
        "perfs": ratings,
        "counts": counts,
    }, 200


def _fetch_profile(platform: str, username: str):
    platform = _normalize_platform(platform)
    if platform == "chesscom":
        return _chesscom_profile(username)
    return _lichess_profile(username)


def _activity_is_recent(last_seen_at: int | None, now: datetime | None = None) -> bool:
    if not last_seen_at:
        return True
    now = now or _now()
    age_ms = now.timestamp() * 1000 - int(last_seen_at)
    return age_ms <= MAX_ACTIVITY_AGE_DAYS * 24 * 60 * 60 * 1000


def select_profile_rating(profile: dict[str, Any]) -> dict[str, Any]:
    """Выбирает рейтинг для стартовой оценки по существующим правилам проекта."""
    platform = _normalize_platform(str(profile.get("platform") or "lichess"))
    perfs = profile.get("perfs") or {}
    activity_known = bool(profile.get("last_seen_at"))
    active = _activity_is_recent(profile.get("last_seen_at"))

    candidates: list[dict[str, Any]] = []
    for rating_type in RATING_PRIORITY:
        perf = perfs.get(rating_type) or {}
        rating = perf.get("rating")
        if rating is None:
            continue

        games = int(perf.get("games") or 0)
        if games == 0 and rating is not None:
            # Аккаунт без партий: даже при наличии поля rating не завышаем
            # стартовую оценку выше NO_GAMES_RATING_CAP.
            rating = min(int(rating), NO_GAMES_RATING_CAP)
        rd = perf.get("rd")
        provisional = bool(perf.get("provisional", False))
        reasons: list[str] = []
        if games == 0:
            reasons.append("0 партий на аккаунте")
        elif games < MIN_GAMES_FOR_ASSESSMENT:
            reasons.append(f"меньше {MIN_GAMES_FOR_ASSESSMENT} партий")
        if platform == "lichess" and rd is not None and float(rd) >= MAX_LICHESS_RD:
            reasons.append(f"RD {rd} ≥ {MAX_LICHESS_RD}")
        if platform == "lichess" and provisional:
            reasons.append("рейтинг помечен как provisional")
        if not active:
            reasons.append("нет активности за последний год")

        candidates.append(
            {
                "rating_type": rating_type,
                "rating": int(rating),
                "games": games,
                "rd": float(rd) if rd is not None else None,
                "provisional": provisional,
                "usable": not reasons,
                "reasons": reasons,
            }
        )

    selected = next((item for item in candidates if item["usable"]), None)
    fallback = candidates[0] if candidates else None
    chosen = selected or fallback

    warnings: list[str] = []
    if not activity_known:
        warnings.append("Публичный API не сообщил дату последней активности; проверка активности пропущена.")
    if chosen and not chosen["usable"]:
        warnings.append(
            "Аккаунт можно привязать, но его рейтинг не будет использован для стартовой оценки: "
            + ", ".join(chosen["reasons"])
            + "."
        )
    if not chosen:
        warnings.append("В профиле не найден рейтинг blitz/rapid/bullet.")

    return {
        "rating_type": chosen["rating_type"] if chosen else None,
        "rating": chosen["rating"] if chosen else None,
        "games": chosen["games"] if chosen else 0,
        "rd": chosen["rd"] if chosen else None,
        "provisional": chosen["provisional"] if chosen else False,
        "usable": bool(chosen and chosen["usable"]),
        "scale": f"{platform}_{chosen['rating_type']}" if chosen else None,
        "candidates": candidates,
        "warnings": warnings,
    }


def apply_public_profile(account: UserChessAccount, profile: dict[str, Any]) -> dict[str, Any]:
    """Обновляет публичные данные, не меняя факт подтверждения владения."""
    rating_info = select_profile_rating(profile)
    account.username = str(profile.get("username") or account.username)
    account.rating_type = rating_info.get("rating_type")
    account.rating = rating_info.get("rating")
    account.rating_scale = rating_info.get("scale")
    account.games = int(rating_info.get("games") or 0)
    account.rating_deviation = rating_info.get("rd")
    account.provisional = bool(rating_info.get("provisional", False))
    account.rating_usable = bool(rating_info.get("usable", False))
    account.profile_snapshot = profile
    account.updated_at = _now()
    if account.linked_at is None:
        account.linked_at = _now()
    return rating_info


def apply_trusted_rating_to_user(user: User, account: UserChessAccount) -> None:
    """Синхронизирует trusted rating с профилем пользователя после OAuth/sync."""
    if not account.rating_trusted:
        return
    user.rating_estimate = int(account.rating)
    user.rating_scale = account.rating_scale

    rating = int(account.rating)
    if rating >= 1800:
        band = 4
    elif rating >= 1400:
        band = 3
    elif rating >= 1000:
        band = 2
    else:
        band = 1
    user.prior_band = band

    # Если анкета уже была заполнена до верификации, обновляем её derived-поля,
    # не меняя ответы пользователя на остальные вопросы.
    if isinstance(user.onboarding, dict):
        onboarding = dict(user.onboarding)
        answers = dict(onboarding.get("answers") or {})
        q2 = dict(answers.get("q2") or {})
        q2.update(
            {
                "platform": account.platform,
                "login": account.username,
                "time_control": account.rating_type or q2.get("time_control") or "blitz",
                "rating": account.rating,
                "rating_used_for_assessment": account.rating,
                "verified": True,
                "verification_status": VERIFICATION_VERIFIED,
                "verification_method": account.verification_method,
                "rating_trusted": True,
            }
        )
        answers["q2"] = q2
        onboarding["answers"] = answers
        onboarding["prior_band"] = band
        user.onboarding = onboarding


def _effective_status(account: UserChessAccount) -> str:
    status = account.verification_status or (VERIFICATION_VERIFIED if account.verified else VERIFICATION_LINKED)
    expires = _utc(account.verification_expires_at)
    if status == VERIFICATION_PENDING and expires and expires <= _now():
        return VERIFICATION_EXPIRED
    return status


def _public_account(account: UserChessAccount) -> dict[str, Any]:
    status = _effective_status(account)
    ownership_verified = bool(account.verified and status == VERIFICATION_VERIFIED)
    rating_trusted = bool(ownership_verified and account.rating_usable and account.rating is not None)
    return {
        "id": account.id,
        "platform": account.platform,
        "username": account.username,
        "external_user_id": account.external_user_id,
        "rating_type": account.rating_type,
        "rating": account.rating,
        "rating_scale": account.rating_scale,
        "games": account.games,
        "rating_deviation": account.rating_deviation,
        "provisional": account.provisional,
        "rating_usable": account.rating_usable,
        "rating_trusted": rating_trusted,
        "verified": ownership_verified,
        "verification_status": status,
        "verification_method": account.verification_method,
        "verified_at": account.verified_at.isoformat() if account.verified_at else None,
        "last_verified_at": account.last_verified_at.isoformat() if account.last_verified_at else None,
        "verification_expires_at": (
            account.verification_expires_at.isoformat() if account.verification_expires_at else None
        ),
        "verification_failure_reason": account.verification_failure_reason,
        "linked_at": account.linked_at.isoformat() if account.linked_at else None,
        "updated_at": account.updated_at.isoformat() if account.updated_at else None,
        "trust": {
            "ownership": "verified" if ownership_verified else "unverified",
            "rating": "trusted" if rating_trusted else "untrusted",
        },
    }


def _public_verification(row: ChessAccountVerification) -> dict[str, Any]:
    return {
        "id": row.id,
        "account_id": row.chess_account_id,
        "platform": row.platform,
        "method": row.method,
        "status": row.status,
        "expected_username": row.expected_username,
        "external_username": row.external_username,
        "external_user_id": row.external_user_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "failure_reason": row.failure_reason,
    }


class LinkChessAccountRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    platform: Literal["lichess", "chesscom"] = "lichess"


@router.get("/api/chess-profile")
def chess_profile(
    username: str = Query(..., min_length=1, max_length=64),
    platform: str = Query("lichess"),
):
    platform = _normalize_platform(platform)
    payload, status = _fetch_profile(platform, username)
    if status == 200:
        payload["assessment_rating"] = select_profile_rating(payload)
    return JSONResponse(content=payload, status_code=status)


@router.get("/api/chess-profile/linked")
async def linked_chess_accounts(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    accounts = (
        await db.scalars(
            select(UserChessAccount)
            .where(UserChessAccount.user_id == user.id)
            .order_by(UserChessAccount.id.asc())
        )
    ).all()

    changed = False
    for account in accounts:
        if (
            account.verification_status == VERIFICATION_PENDING
            and _utc(account.verification_expires_at)
            and _utc(account.verification_expires_at) <= _now()
            and not account.verified
        ):
            account.verification_status = VERIFICATION_EXPIRED
            account.verification_failure_reason = "Срок OAuth-проверки истёк"
            changed = True
    if changed:
        await db.commit()

    return {"items": [_public_account(account) for account in accounts]}


@router.post("/api/chess-profile/link")
async def link_chess_account(
    req: LinkChessAccountRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Находит публичный профиль и сохраняет soft-link за текущим пользователем."""
    username = req.username.strip()
    platform = _normalize_platform(req.platform)
    payload, status = await run_in_threadpool(_fetch_profile, platform, username)
    if status != 200:
        raise HTTPException(status_code=status, detail=payload.get("error") or "Профиль не найден")

    canonical_username = str(payload.get("username") or username)
    account = await db.scalar(
        select(UserChessAccount)
        .where(UserChessAccount.user_id == user.id, UserChessAccount.platform == platform)
        .with_for_update()
    )
    if account is None:
        account = UserChessAccount(user_id=user.id, platform=platform, username=canonical_username)
        db.add(account)
    elif account.verified and account.username.casefold() != canonical_username.casefold():
        raise HTTPException(
            status_code=409,
            detail="На этой платформе уже подтверждён другой аккаунт. Сначала отключите его.",
        )

    rating_info = apply_public_profile(account, payload)
    if not account.verified:
        account.external_user_id = None
        account.verified_at = None
        account.last_verified_at = None
        account.verification_status = VERIFICATION_LINKED
        account.verification_method = None
        account.verification_expires_at = None
        account.verification_failure_reason = None

    await db.commit()
    await db.refresh(account)
    return {
        "ok": True,
        "account": _public_account(account),
        "profile": payload,
        "assessment_rating": rating_info,
        "ownership_verified": account.ownership_verified,
        "notice": (
            "Аккаунт подтверждён OAuth."
            if account.ownership_verified
            else "Профиль добавлен, но владение аккаунтом ещё не подтверждено."
        ),
    }


@router.post("/api/chess-profile/sync/{platform}")
async def sync_chess_account(
    platform: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    platform = _normalize_platform(platform)
    account = await db.scalar(
        select(UserChessAccount).where(
            UserChessAccount.user_id == user.id,
            UserChessAccount.platform == platform,
        )
    )
    if account is None:
        raise HTTPException(status_code=404, detail="Шахматный аккаунт не подключён")

    payload, status = await run_in_threadpool(_fetch_profile, platform, account.username)
    if status != 200:
        raise HTTPException(status_code=status, detail=payload.get("error") or "Не удалось обновить профиль")
    apply_public_profile(account, payload)
    apply_trusted_rating_to_user(user, account)
    await db.commit()
    await db.refresh(account)
    return {"ok": True, "account": _public_account(account)}


@router.get("/api/chess-profile/verifications/{platform}")
async def chess_account_verification_history(
    platform: str,
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    platform = _normalize_platform(platform)
    rows = (
        await db.scalars(
            select(ChessAccountVerification)
            .where(
                ChessAccountVerification.user_id == user.id,
                ChessAccountVerification.platform == platform,
            )
            .order_by(ChessAccountVerification.id.desc())
            .limit(limit)
        )
    ).all()
    return {"items": [_public_verification(row) for row in rows]}


@router.delete("/api/chess-profile/link/{platform}")
async def unlink_chess_account(
    platform: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    platform = _normalize_platform(platform)
    account = await db.scalar(
        select(UserChessAccount).where(
            UserChessAccount.user_id == user.id,
            UserChessAccount.platform == platform,
        )
    )
    if account is None:
        return {"ok": True, "deleted": False}

    now = _now()
    db.add(
        ChessAccountVerification(
            chess_account_id=account.id,
            user_id=user.id,
            platform=platform,
            method=VERIFICATION_METHOD_USER_ACTION,
            status=VERIFICATION_REVOKED,
            expected_username=account.username,
            external_username=account.username,
            external_user_id=account.external_user_id,
            started_at=now,
            completed_at=now,
            details={"reason": "user_unlinked_account"},
        )
    )
    await db.flush()
    await db.delete(account)
    await db.commit()
    return {"ok": True, "deleted": True}
