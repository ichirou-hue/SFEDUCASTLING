"""Заголовки безопасности для всех ответов приложения."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.config.settings import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Добавляет HSTS, CSP, X-Frame-Options и другие защитные заголовки.

    CSP сознательно разрешает inline-скрипты/стили: в собранном SPA есть
    небольшой inline-скрипт инициализации темы. Если позже появится nonce,
    политику можно ужесточить через CONTENT_SECURITY_POLICY.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if not settings.security.security_headers_enabled:
            return response

        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=()",
        )
        headers.setdefault(
            "Content-Security-Policy",
            settings.security.content_security_policy,
        )

        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        )
        if is_https:
            headers.setdefault(
                "Strict-Transport-Security",
                f"max-age={settings.security.hsts_max_age}; includeSubDomains",
            )

        # API-ответы и HTML не кэшируем: там могут быть персональные данные.
        content_type = headers.get("content-type", "")
        if request.url.path.startswith("/api") or "text/html" in content_type:
            headers.setdefault("Cache-Control", "no-store")
        return response
