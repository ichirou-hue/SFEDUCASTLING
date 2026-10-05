"""Профиль и мягкая привязка внешнего шахматного аккаунта.

Публичные API:
  - Lichess:   GET https://lichess.org/api/user/{username}
  - Chess.com: GET https://api.chess.com/pub/player/{username}
               GET https://api.chess.com/pub/player/{username}/stats

Привязка в этом модуле является *soft link*: пользователь подтверждает, что
публичный профиль принадлежит ему, но владение аккаунтом не доказывается OAuth.
Это намеренно хранится в поле ``verified=False`` до появления настоящей OAuth-
верификации.
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
from backend.db.session import get_db
from backend.models.user import User
from backend.models.user_chess_account import UserChessAccount

router = APIRouter(tags=["chess-profile"])

LICHESS_USER_URL = "https://lichess.org/api/user/{username}"
CHESSCOM_PLAYER_URL = "https://api.chess.com/pub/player/{username}"
CHESSCOM_STATS_URL = "https://api.chess.com/pub/player/{username}/stats"

USER_AGENT = "SfeduCastling/0.2 (linked-chess-profile)"
HTTP_TIMEOUT = 8.0
RATING_PRIORITY = ("blitz", "rapid", "bullet")
MIN_GAMES_FOR_ASSESSMENT = 30
MAX_LICHESS_RD = 110
MAX_ACTIVITY_AGE_DAYS = 365


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
        "last_seen_at": data.get("seenAt"),  # unix ms, если Lichess его отдаёт
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
            "rating": last.get("rating"),
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


def _activity_is_recent(last_seen_at: int | None, now: datetime | None = None) -> bool:
    if not last_seen_at:
        # Не все публичные профили отдают last seen. Отсутствие поля само по себе
        # не делает рейтинг недостоверным; этот факт отдельно показывается в warning.
        return True
    now = now or datetime.now(UTC)
    age_ms = now.timestamp() * 1000 - int(last_seen_at)
    return age_ms <= MAX_ACTIVITY_AGE_DAYS * 24 * 60 * 60 * 1000


def select_profile_rating(profile: dict[str, Any]) -> dict[str, Any]:
    """Выбирает рейтинг для стартовой оценки по правилам анкеты.

    Приоритет внутри платформы: blitz -> rapid -> bullet. Для использования
    в level-test рейтинг должен иметь >=30 партий; для Lichess дополнительно
    RD < 110 и отсутствие provisional-флага. Если доступна дата активности,
    профиль должен быть активен за последний год.
    """
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
        rd = perf.get("rd")
        provisional = bool(perf.get("provisional", False))
        reasons: list[str] = []
        if games < MIN_GAMES_FOR_ASSESSMENT:
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


def _public_account(account: UserChessAccount) -> dict[str, Any]:
    return {
        "id": account.id,
        "platform": account.platform,
        "username": account.username,
        "rating_type": account.rating_type,
        "rating": account.rating,
        "rating_scale": account.rating_scale,
        "games": account.games,
        "rating_deviation": account.rating_deviation,
        "provisional": account.provisional,
        "rating_usable": account.rating_usable,
        "verified": account.verified,
        "linked_at": account.linked_at.isoformat() if account.linked_at else None,
        "updated_at": account.updated_at.isoformat() if account.updated_at else None,
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
    if platform == "chesscom":
        payload, status = _chesscom_profile(username)
    else:
        payload, status = _lichess_profile(username)

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
    return {"items": [_public_account(account) for account in accounts]}


@router.post("/api/chess-profile/link")
async def link_chess_account(
    req: LinkChessAccountRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Находит публичный профиль и сохраняет soft link за текущим пользователем."""
    username = req.username.strip()
    platform = _normalize_platform(req.platform)

    if platform == "chesscom":
        payload, status = await run_in_threadpool(_chesscom_profile, username)
    else:
        payload, status = await run_in_threadpool(_lichess_profile, username)

    if status != 200:
        raise HTTPException(status_code=status, detail=payload.get("error") or "Профиль не найден")

    rating_info = select_profile_rating(payload)
    canonical_username = str(payload.get("username") or username)
    now = datetime.now(UTC)

    account = await db.scalar(
        select(UserChessAccount)
        .where(
            UserChessAccount.user_id == user.id,
            UserChessAccount.platform == platform,
        )
        .with_for_update()
    )
    if account is None:
        account = UserChessAccount(user_id=user.id, platform=platform, username=canonical_username)
        db.add(account)

    account.username = canonical_username
    account.rating_type = rating_info.get("rating_type")
    account.rating = rating_info.get("rating")
    account.rating_scale = rating_info.get("scale")
    account.games = int(rating_info.get("games") or 0)
    account.rating_deviation = rating_info.get("rd")
    account.provisional = bool(rating_info.get("provisional", False))
    account.rating_usable = bool(rating_info.get("usable", False))
    # Без OAuth подтверждение владения невозможно.
    account.verified = False
    account.profile_snapshot = payload
    account.updated_at = now
    if account.linked_at is None:
        account.linked_at = now

    await db.commit()
    await db.refresh(account)

    return {
        "ok": True,
        "account": _public_account(account),
        "profile": payload,
        "assessment_rating": rating_info,
        "ownership_verified": False,
        "notice": "Аккаунт привязан по публичному профилю. Владение аккаунтом не подтверждено OAuth.",
    }


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
    await db.delete(account)
    await db.commit()
    return {"ok": True, "deleted": True}
