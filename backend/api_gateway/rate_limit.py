"""Минимальный in-memory лимитер частоты запросов.

Рассчитан на один процесс uvicorn: для нескольких воркеров или нескольких
машин нужен внешний счётчик (Redis). Здесь важно закрыть перебор паролей и
дорогие эндпоинты без новых зависимостей.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import deque
from collections.abc import Callable

import jwt
from fastapi import HTTPException, Request

from backend.api_gateway.security import decode_access_token
from backend.config.settings import settings


def _is_trusted_proxy(host: str) -> bool:
    """Проверяет, что непосредственный клиент - частный или loopback адрес.

    Нельзя проверять префиксами: ``172.2`` ошибочно считался бы частным
    для публичных адресов вроде ``172.200.x.x``, и тогда клиент мог бы
    подделать ``X-Forwarded-For``, обойдя лимит частоты.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback


class SlidingWindowLimiter:
    """Sliding window по ключу (bucket, identity)."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def hit(self, bucket: str, identity: str, limit: int, window_sec: int) -> tuple[bool, int]:
        """Регистрирует попытку.

        Returns:
            (allowed, retry_after_sec)
        """
        if limit <= 0 or window_sec <= 0:
            return True, 0

        now = time.monotonic()
        cutoff = now - window_sec
        key = (bucket, identity)
        with self._lock:
            window = self._hits.setdefault(key, deque())
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= limit:
                retry_after = max(1, int(window[0] + window_sec - now) + 1)
                return False, retry_after
            window.append(now)
        return True, 0


LIMITER = SlidingWindowLimiter()


def _client_ip(request: Request) -> str:
    client_host = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded and _is_trusted_proxy(client_host):
        candidate = forwarded.split(",", 1)[0].strip()
        if candidate:
            return candidate
    return client_host or "unknown"


def _identity(request: Request, *, by_user: bool) -> str:
    """Ключ лимита.

    Для входа и регистрации ключ всегда IP: иначе случайный заголовок
    ``Authorization`` создавал бы новый bucket и обходил защиту от перебора.
    Для остальных endpoint'ов валидный access-токен даёт ключ пользователя,
    а невалидный токен снова считается по IP.
    """
    if by_user:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            try:
                payload = decode_access_token(authorization[7:])
            except jwt.PyJWTError:
                pass
            else:
                subject = payload.get("sub")
                if subject:
                    return f"user:{subject}"
    return f"ip:{_client_ip(request)}"


def rate_limit(
    bucket: str,
    limit_getter: Callable[[], int],
    window_getter: Callable[[], int],
    *,
    by_user: bool = False,
) -> Callable:
    """FastAPI-dependency. Значения читаются на каждом запросе из настроек."""

    async def _dependency(request: Request) -> None:
        if not settings.rate_limit.enabled:
            return
        allowed, retry_after = LIMITER.hit(
            bucket,
            _identity(request, by_user=by_user),
            limit_getter(),
            window_getter(),
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Слишком много запросов. Попробуйте позже.",
                headers={"Retry-After": str(retry_after)},
            )

    return _dependency


login_rate_limit = rate_limit(
    "login",
    lambda: settings.rate_limit.login_attempts,
    lambda: settings.rate_limit.login_window_sec,
)
auth_rate_limit = rate_limit(
    "auth",
    lambda: settings.rate_limit.auth_attempts,
    lambda: settings.rate_limit.auth_window_sec,
)
chat_rate_limit = rate_limit(
    "chat",
    lambda: settings.rate_limit.chat_attempts,
    lambda: settings.rate_limit.chat_window_sec,
    by_user=True,
)
expensive_rate_limit = rate_limit(
    "expensive",
    lambda: settings.rate_limit.expensive_attempts,
    lambda: settings.rate_limit.expensive_window_sec,
    by_user=True,
)
