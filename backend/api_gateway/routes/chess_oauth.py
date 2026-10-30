"""OAuth-привязка, подтверждение владения и вход через шахматные аккаунты.

Lichess: Authorization Code + PKCE (S256), client_secret не нужен.
Chess.com: OAuth-контур активируется после выдачи developer credentials.

Внешние access token'ы намеренно не сохраняются: они используются только внутри
callback, чтобы доказать владение аккаунтом и получить идентичность пользователя.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit

import jwt
import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.api_gateway.dependecies import get_optional_current_user
from backend.api_gateway.routes.auth import _issue_tokens
from backend.api_gateway.routes.chess_profile import (
    _chesscom_profile,
    _lichess_profile,
    apply_public_profile,
    apply_trusted_rating_to_user,
    select_profile_rating,
)
from backend.api_gateway.security import hash_password
from backend.config.settings import settings
from backend.db.session import get_db
from backend.models.user import ROLE_LEARNER, User
from backend.models.user_chess_account import (
    ChessAccountVerification,
    UserChessAccount,
    VERIFICATION_ERROR,
    VERIFICATION_EXPIRED,
    VERIFICATION_LINKED,
    VERIFICATION_METHOD_OAUTH,
    VERIFICATION_PENDING,
    VERIFICATION_VERIFIED,
)

router = APIRouter(prefix="/api/auth/oauth", tags=["chess-oauth"])

FLOW_COOKIE = "gigachess_oauth_flow"
RESULT_COOKIE = "gigachess_oauth_result"

LICHESS_AUTHORIZE_URL = "https://lichess.org/oauth"
LICHESS_TOKEN_URL = "https://lichess.org/api/token"
LICHESS_ACCOUNT_URL = "https://lichess.org/api/account"


class OAuthStartRequest(BaseModel):
    intent: Literal["login", "link", "verify"] = "login"
    return_to: str = Field(default="/", max_length=512)
    popup: bool = True


def _now() -> datetime:
    return datetime.now(UTC)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _pkce_pair() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    if value in {"chess.com", "chess_com", "chess", "chesscom"}:
        return "chesscom"
    if value == "lichess":
        return "lichess"
    raise HTTPException(status_code=404, detail="Неизвестный OAuth-провайдер")


def _safe_return_to(value: str | None) -> str:
    path = (value or "/").strip()
    if not path.startswith("/") or path.startswith("//"):
        return "/"
    return path


def _public_base_url(request: Request) -> str:
    configured = settings.chess_oauth.public_base_url.strip()
    return configured.rstrip("/") if configured else str(request.base_url).rstrip("/")


def _frontend_base_url(request: Request) -> str:
    configured = settings.chess_oauth.frontend_base_url.strip()
    return configured.rstrip("/") if configured else _public_base_url(request)


def _callback_url(request: Request, provider: str) -> str:
    return f"{_public_base_url(request)}/api/auth/oauth/{provider}/callback"


def _encode_flow(payload: dict[str, Any]) -> str:
    now = _now()
    body = {
        **payload,
        "type": "chess_oauth_flow",
        "iat": now,
        "exp": now + timedelta(seconds=settings.chess_oauth.flow_ttl_sec),
    }
    return jwt.encode(body, settings.auth.jwt_secret, algorithm="HS256")


def _decode_signed_cookie(raw: str | None, expected_type: str) -> dict[str, Any]:
    if not raw:
        raise HTTPException(status_code=400, detail="OAuth-сессия не найдена или истекла")
    try:
        payload = jwt.decode(raw, settings.auth.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=400, detail="OAuth-сессия недействительна") from exc
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=400, detail="Некорректный тип OAuth-сессии")
    return payload


def _cookie_kwargs() -> dict[str, Any]:
    return {
        "httponly": True,
        "secure": settings.chess_oauth.cookie_secure,
        "samesite": "lax",
        "path": "/",
        "max_age": settings.chess_oauth.flow_ttl_sec,
    }


def _provider_capabilities() -> dict[str, Any]:
    cfg = settings.chess_oauth
    chesscom_enabled = bool(
        cfg.chesscom_client_id.strip()
        and cfg.chesscom_authorize_url.strip()
        and cfg.chesscom_token_url.strip()
        and cfg.chesscom_userinfo_url.strip()
    )
    return {
        "lichess": {
            "oauth": True,
            "verification": True,
            "pkce": True,
            "public_link": True,
        },
        "chesscom": {
            "oauth": chesscom_enabled,
            "verification": chesscom_enabled,
            "pkce": bool(cfg.chesscom_use_pkce) if chesscom_enabled else False,
            "public_link": True,
            "requires_developer_access": not chesscom_enabled,
        },
    }


def _build_authorization_url(
    provider: str,
    *,
    state: str,
    verifier: str,
    challenge: str,
    redirect_uri: str,
) -> str:
    if provider == "lichess":
        params = {
            "response_type": "code",
            "client_id": settings.chess_oauth.lichess_client_id,
            "redirect_uri": redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
            "scope": "",
        }
        return f"{LICHESS_AUTHORIZE_URL}?{urlencode(params)}"

    cfg = settings.chess_oauth
    if not _provider_capabilities()["chesscom"]["oauth"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Chess.com OAuth пока не настроен. Получите OAuth-доступ в Chess.com "
                "Developer Community и заполните CHESSCOM_OAUTH_* в .env."
            ),
        )

    params: dict[str, str] = {
        "response_type": "code",
        "client_id": cfg.chesscom_client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if cfg.chesscom_scope.strip():
        params["scope"] = cfg.chesscom_scope.strip()
    if cfg.chesscom_use_pkce:
        params["code_challenge_method"] = "S256"
        params["code_challenge"] = challenge
    return f"{cfg.chesscom_authorize_url.strip()}?{urlencode(params)}"


def _token_exchange(provider: str, code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
    if provider == "lichess":
        resp = requests.post(
            LICHESS_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": settings.chess_oauth.lichess_client_id,
                "code_verifier": verifier,
            },
            headers={"Accept": "application/json"},
            timeout=settings.chess_oauth.http_timeout,
        )
    else:
        cfg = settings.chess_oauth
        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": cfg.chesscom_client_id,
        }
        if cfg.chesscom_use_pkce:
            data["code_verifier"] = verifier
        auth = None
        if cfg.chesscom_client_secret:
            if cfg.chesscom_token_auth_method == "client_secret_basic":
                auth = (cfg.chesscom_client_id, cfg.chesscom_client_secret)
            else:
                data["client_secret"] = cfg.chesscom_client_secret
        resp = requests.post(
            cfg.chesscom_token_url.strip(),
            data=data,
            auth=auth,
            headers={"Accept": "application/json"},
            timeout=cfg.http_timeout,
        )

    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not 200 <= resp.status_code < 300:
        detail = payload.get("error_description") or payload.get("error") or f"HTTP {resp.status_code}"
        raise RuntimeError(f"OAuth token exchange: {detail}")
    if not payload.get("access_token"):
        raise RuntimeError("OAuth-провайдер не вернул access_token")
    return payload


def _oauth_identity(provider: str, access_token: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": settings.chess_oauth.user_agent,
    }

    if provider == "lichess":
        resp = requests.get(LICHESS_ACCOUNT_URL, headers=headers, timeout=settings.chess_oauth.http_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Lichess /api/account вернул HTTP {resp.status_code}")
        identity = resp.json()
        username = str(identity.get("username") or identity.get("id") or "").strip()
        external_id = str(identity.get("id") or username.lower()).strip()
        if not username:
            raise RuntimeError("Lichess не вернул username")
        profile, status = _lichess_profile(username)
    else:
        cfg = settings.chess_oauth
        resp = requests.get(cfg.chesscom_userinfo_url.strip(), headers=headers, timeout=cfg.http_timeout)
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(f"Chess.com userinfo вернул HTTP {resp.status_code}")
        identity = resp.json()
        user_block = identity.get("user") if isinstance(identity.get("user"), dict) else {}
        username = str(
            identity.get("username") or identity.get("user_name") or user_block.get("username") or ""
        ).strip()
        external_id = str(
            identity.get("id")
            or identity.get("player_id")
            or identity.get("uuid")
            or identity.get("sub")
            or username.lower()
        ).strip()
        if not username:
            raise RuntimeError(
                "Chess.com userinfo не содержит username. Укажите корректный CHESSCOM_OAUTH_USERINFO_URL."
            )
        profile, status = _chesscom_profile(username)

    if status != 200:
        raise RuntimeError(profile.get("error") or "Не удалось получить публичный шахматный профиль")
    return identity, profile, external_id


def _oauth_exchange_and_profile(
    provider: str,
    code: str,
    verifier: str,
    redirect_uri: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    token_payload = _token_exchange(provider, code, verifier, redirect_uri)
    return _oauth_identity(provider, str(token_payload["access_token"]))


def _set_verified_identity(
    account: UserChessAccount,
    profile: dict[str, Any],
    external_user_id: str,
) -> None:
    now = _now()
    apply_public_profile(account, profile)
    account.verified = True
    account.external_user_id = external_user_id
    account.verification_status = VERIFICATION_VERIFIED
    account.verification_method = VERIFICATION_METHOD_OAUTH
    if account.verified_at is None:
        account.verified_at = now
    account.last_verified_at = now
    account.verification_expires_at = None
    account.verification_failure_reason = None


async def _verified_link(
    db: AsyncSession,
    user: User,
    provider: str,
    profile: dict[str, Any],
    external_user_id: str,
) -> UserChessAccount:
    if external_user_id:
        owner = await db.scalar(
            select(UserChessAccount).where(
                UserChessAccount.platform == provider,
                UserChessAccount.external_user_id == external_user_id,
                UserChessAccount.verified.is_(True),
            )
        )
        if owner is not None and owner.user_id != user.id:
            raise HTTPException(
                status_code=409,
                detail="Этот шахматный аккаунт уже подтверждён другим пользователем GigaChess",
            )

    account = await db.scalar(
        select(UserChessAccount).where(
            UserChessAccount.user_id == user.id,
            UserChessAccount.platform == provider,
        )
    )
    if account is None:
        account = UserChessAccount(
            user_id=user.id,
            platform=provider,
            username=str(profile.get("username") or ""),
            verification_status=VERIFICATION_LINKED,
        )
        db.add(account)
        await db.flush()

    _set_verified_identity(account, profile, external_user_id)
    await db.flush()
    return account


async def _verification_row(db: AsyncSession, verification_id: Any) -> ChessAccountVerification | None:
    try:
        row_id = int(verification_id)
    except (TypeError, ValueError):
        return None
    return await db.get(ChessAccountVerification, row_id)


async def _mark_verification_success(
    db: AsyncSession,
    flow: dict[str, Any],
    account: UserChessAccount,
    external_user_id: str,
) -> None:
    row = await _verification_row(db, flow.get("verification_id"))
    now = _now()
    if row is None:
        row = ChessAccountVerification(
            chess_account_id=account.id,
            user_id=account.user_id,
            platform=account.platform,
            method=VERIFICATION_METHOD_OAUTH,
            status=VERIFICATION_VERIFIED,
            expected_username=flow.get("expected_username"),
            started_at=now,
        )
        db.add(row)
    row.chess_account_id = account.id
    row.status = VERIFICATION_VERIFIED
    row.external_username = account.username
    row.external_user_id = external_user_id
    row.completed_at = now
    row.failure_reason = None
    row.details = {"intent": flow.get("intent")}


async def _mark_verification_failure(
    db: AsyncSession,
    flow: dict[str, Any],
    provider: str,
    reason: str,
    *,
    status: str = VERIFICATION_ERROR,
) -> None:
    row = await _verification_row(db, flow.get("verification_id"))
    now = _now()
    if row is not None:
        row.status = status
        row.completed_at = now
        row.failure_reason = reason[:255]

    user_id = flow.get("user_id")
    if user_id is not None:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            uid = None
        if uid is not None:
            account = await db.scalar(
                select(UserChessAccount).where(
                    UserChessAccount.user_id == uid,
                    UserChessAccount.platform == provider,
                )
            )
            # Не снимаем уже доказанное владение из-за неудачной повторной попытки.
            if account is not None and not account.verified:
                account.verification_status = status
                account.verification_failure_reason = reason[:255]
                account.verification_expires_at = None
    await db.commit()


def _slug_login(username: str, provider: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", username).strip("_-")
    if len(base) < 3:
        base = f"{provider}_user"
    return base[:32]


async def _unique_login(db: AsyncSession, username: str, provider: str) -> str:
    base = _slug_login(username, provider)
    candidate = base
    suffix_base = f"_{provider}"
    n = 0
    while await db.scalar(select(User.id).where(func.lower(User.login) == candidate.lower())):
        n += 1
        suffix = suffix_base if n == 1 else f"{suffix_base}{n}"
        candidate = f"{base[: max(3, 32 - len(suffix))]}{suffix}"
    return candidate


async def _login_or_create_from_oauth(
    db: AsyncSession,
    provider: str,
    profile: dict[str, Any],
    external_user_id: str,
) -> tuple[User, UserChessAccount]:
    linked = None
    if external_user_id:
        linked = await db.scalar(
            select(UserChessAccount).where(
                UserChessAccount.platform == provider,
                UserChessAccount.external_user_id == external_user_id,
                UserChessAccount.verified.is_(True),
            )
        )
    if linked is not None:
        user = await db.get(User, linked.user_id)
        if user is None:
            raise HTTPException(status_code=409, detail="Связанный пользователь GigaChess не найден")
        _set_verified_identity(linked, profile, external_user_id)
        return user, linked

    username = str(profile.get("username") or provider)
    login = await _unique_login(db, username, provider)
    user = User(
        login=login,
        email=None,
        elo=None,
        password_hash=hash_password(secrets.token_urlsafe(32)),
        role=ROLE_LEARNER,
    )
    db.add(user)
    await db.flush()
    account = await _verified_link(db, user, provider, profile, external_user_id)

    rating_info = select_profile_rating(profile)
    if account.rating_trusted and rating_info.get("rating") is not None:
        apply_trusted_rating_to_user(user, account)
    return user, account


def _popup_html(payload: dict[str, Any], target_origin: str) -> HTMLResponse:
    message_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    parts = urlsplit(target_origin)
    normalized_origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else target_origin
    origin_json = json.dumps(normalized_origin)
    fallback_text = html.escape(payload.get("message") or "OAuth завершён")
    body = f"""<!doctype html>
<html lang=\"ru\"><head><meta charset=\"utf-8\"><title>GigaChess OAuth</title></head>
<body style=\"font-family:system-ui,sans-serif;padding:24px\">
<p>{fallback_text}</p>
<script>
(function() {{
  var message = {message_json};
  if (window.opener && !window.opener.closed) {{
    window.opener.postMessage(message, {origin_json});
    window.close();
  }}
}})();
</script>
</body></html>"""
    return HTMLResponse(body)


def _finish_browser_response(
    request: Request,
    flow: dict[str, Any],
    *,
    provider: str,
    success: bool,
    message: str,
    result_cookie: str | None = None,
):
    frontend = _frontend_base_url(request)
    intent = str(flow.get("intent") or "login")
    payload = {
        "type": "gigachess-oauth",
        "provider": provider,
        "intent": intent,
        "status": "success" if success else "error",
        "message": message,
    }

    if bool(flow.get("popup")):
        response = _popup_html(payload, frontend)
    else:
        if success and intent in {"link", "verify"}:
            return_to = _safe_return_to(flow.get("return_to"))
            separator = "&" if "?" in return_to else "?"
            target = f"{frontend}{return_to}{separator}oauth_link=success&provider={provider}"
        else:
            params = urlencode(
                {
                    "provider": provider,
                    "status": "success" if success else "error",
                    "message": message,
                }
            )
            target = f"{frontend}/oauth/callback?{params}"
        response = RedirectResponse(target, status_code=302)

    response.delete_cookie(FLOW_COOKIE, path="/")
    if result_cookie:
        response.set_cookie(RESULT_COOKIE, result_cookie, **_cookie_kwargs())
    return response


@router.get("/providers")
async def oauth_providers():
    return _provider_capabilities()


@router.post("/{provider}/start")
async def oauth_start(
    provider: str,
    req: OAuthStartRequest,
    request: Request,
    user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
):
    provider = _normalize_provider(provider)
    if req.intent in {"link", "verify"} and user is None:
        raise HTTPException(status_code=401, detail="Для привязки аккаунта требуется войти в GigaChess")

    account: UserChessAccount | None = None
    expected_username: str | None = None
    if user is not None and req.intent in {"link", "verify"}:
        account = await db.scalar(
            select(UserChessAccount).where(
                UserChessAccount.user_id == user.id,
                UserChessAccount.platform == provider,
            )
        )
        if req.intent == "verify" and account is None:
            raise HTTPException(status_code=404, detail="Сначала добавьте профиль, затем подтвердите владение")
        if account is not None:
            expected_username = account.username

    state = secrets.token_urlsafe(32)
    verifier, challenge = _pkce_pair()
    redirect_uri = _callback_url(request, provider)
    authorization_url = _build_authorization_url(
        provider,
        state=state,
        verifier=verifier,
        challenge=challenge,
        redirect_uri=redirect_uri,
    )

    verification_id: int | None = None
    if user is not None and req.intent in {"link", "verify"}:
        expires_at = _now() + timedelta(seconds=settings.chess_oauth.flow_ttl_sec)
        verification = ChessAccountVerification(
            chess_account_id=account.id if account else None,
            user_id=user.id,
            platform=provider,
            method=VERIFICATION_METHOD_OAUTH,
            status=VERIFICATION_PENDING,
            expected_username=expected_username,
            started_at=_now(),
            expires_at=expires_at,
            details={"intent": req.intent},
        )
        db.add(verification)
        if account is not None and not account.verified:
            account.verification_status = VERIFICATION_PENDING
            account.verification_method = VERIFICATION_METHOD_OAUTH
            account.verification_expires_at = expires_at
            account.verification_failure_reason = None
        await db.commit()
        await db.refresh(verification)
        verification_id = verification.id

    flow_cookie = _encode_flow(
        {
            "provider": provider,
            "state": state,
            "verifier": verifier,
            "intent": req.intent,
            "user_id": user.id if user else None,
            "return_to": _safe_return_to(req.return_to),
            "popup": bool(req.popup),
            "redirect_uri": redirect_uri,
            "verification_id": verification_id,
            "expected_username": expected_username,
        }
    )

    response = JSONResponse(
        {
            "provider": provider,
            "intent": req.intent,
            "authorization_url": authorization_url,
            "verification_id": verification_id,
        }
    )
    response.set_cookie(FLOW_COOKIE, flow_cookie, **_cookie_kwargs())
    return response


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    provider = _normalize_provider(provider)
    flow: dict[str, Any] = {"intent": "login", "popup": False, "return_to": "/"}
    try:
        flow = _decode_signed_cookie(request.cookies.get(FLOW_COOKIE), "chess_oauth_flow")
        if flow.get("provider") != provider or not state or state != flow.get("state"):
            raise HTTPException(status_code=400, detail="OAuth state не совпадает")

        if error:
            message = error_description or error
            await _mark_verification_failure(db, flow, provider, f"Авторизация отменена: {message}")
            return _finish_browser_response(
                request,
                flow,
                provider=provider,
                success=False,
                message=f"Авторизация отменена: {message}",
            )
        if not code:
            raise HTTPException(status_code=400, detail="OAuth-провайдер не вернул authorization code")

        _, profile, external_user_id = await run_in_threadpool(
            _oauth_exchange_and_profile,
            provider,
            code,
            str(flow.get("verifier") or ""),
            str(flow.get("redirect_uri") or _callback_url(request, provider)),
        )

        actual_username = str(profile.get("username") or "").strip()
        expected_username = str(flow.get("expected_username") or "").strip()
        if expected_username and actual_username.casefold() != expected_username.casefold():
            reason = (
                f"В OAuth авторизован аккаунт {actual_username}, а ожидался {expected_username}. "
                "Войдите во внешний аккаунт, который был добавлен в GigaChess."
            )
            await _mark_verification_failure(db, flow, provider, reason)
            return _finish_browser_response(
                request,
                flow,
                provider=provider,
                success=False,
                message=reason,
            )

        if flow.get("intent") in {"link", "verify"}:
            user_id = flow.get("user_id")
            user = await db.get(User, int(user_id)) if user_id is not None else None
            if user is None:
                raise HTTPException(status_code=401, detail="Пользователь для привязки не найден")
            account = await _verified_link(db, user, provider, profile, external_user_id)
            apply_trusted_rating_to_user(user, account)
            await _mark_verification_success(db, flow, account, external_user_id)
            await db.commit()
            return _finish_browser_response(
                request,
                flow,
                provider=provider,
                success=True,
                message=f"{account.username}: владение аккаунтом подтверждено",
            )

        user, account = await _login_or_create_from_oauth(db, provider, profile, external_user_id)
        await _mark_verification_success(db, flow, account, external_user_id)
        await db.commit()
        await db.refresh(user)

        now = _now()
        result_cookie = jwt.encode(
            {
                "type": "chess_oauth_result",
                "sub": str(user.id),
                "provider": provider,
                "iat": now,
                "exp": now + timedelta(seconds=settings.chess_oauth.flow_ttl_sec),
            },
            settings.auth.jwt_secret,
            algorithm="HS256",
        )
        return _finish_browser_response(
            request,
            flow,
            provider=provider,
            success=True,
            message=f"Вход через {provider} выполнен",
            result_cookie=result_cookie,
        )
    except HTTPException as exc:
        await _mark_verification_failure(db, flow, provider, str(exc.detail))
        return _finish_browser_response(
            request,
            flow,
            provider=provider,
            success=False,
            message=str(exc.detail),
        )
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        message = f"Не удалось завершить OAuth: {exc}"
        await _mark_verification_failure(db, flow, provider, message)
        return _finish_browser_response(
            request,
            flow,
            provider=provider,
            success=False,
            message=message,
        )


@router.post("/complete")
async def oauth_complete(request: Request, db: AsyncSession = Depends(get_db)):
    payload = _decode_signed_cookie(request.cookies.get(RESULT_COOKIE), "chess_oauth_result")
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Некорректный результат OAuth") from exc

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь GigaChess не найден")

    tokens = await _issue_tokens(db, user)
    response = JSONResponse({"message": "OAuth-вход завершён", **tokens})
    response.delete_cookie(RESULT_COOKIE, path="/")
    return response
