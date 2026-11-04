"""Проверка безопасности конфигурации при старте приложения.

В production приложение обязано падать, если остались dev-значения:
дефолтный JWT-секрет, открытый CORS, не задан публичный адрес OAuth и т.п.
В development те же проблемы записываются в лог как предупреждения, чтобы не
ломать локальный запуск.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class SecurityConfigError(RuntimeError):
    """Конфигурация небезопасна для production."""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _is_local_origin(origin: str) -> bool:
    """True для loopback/unspecified-источника, включая любой регистр."""
    try:
        hostname = urlsplit(origin.strip()).hostname
    except ValueError:
        return False
    if not hostname:
        return False
    host = hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _is_https_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


def trusted_host_allowed(
    host: str | None,
    trusted_hosts: list[str] | tuple[str, ...] | None,
    *,
    is_production: bool,
) -> bool:
    """Проверяет Host по списку доверенных имён.

    Пустой список в development означает «проверка отключена». В production
    пустой список запрещён отдельной проверкой в ``validate_runtime_settings``.
    Поддерживаются wildcard-домены вида ``*.example.com``.
    """
    hosts = _as_list(trusted_hosts)
    raw_host = (host or "").strip().lower()
    if raw_host.startswith("["):
        hostname = raw_host.split("]", 1)[0].lstrip("[")
    else:
        hostname = raw_host.split(":", 1)[0]
    if not hostname:
        return False
    if not hosts:
        return not is_production
    for allowed in hosts:
        candidate = allowed.strip().lower()
        if not candidate:
            continue
        if candidate == "*":
            return True
        if candidate.startswith("*.") and hostname.endswith(candidate[1:]):
            return True
        if hostname == candidate:
            return True
    return False


def collect_security_issues(settings_obj: Any) -> list[str]:
    """Возвращает список небезопасных мест конфигурации."""
    issues: list[str] = []

    auth = settings_obj.auth
    oauth = settings_obj.chess_oauth
    security = settings_obj.security
    cors = settings_obj.cors

    if auth.uses_default_secret:
        issues.append(
            "JWT_SECRET_KEY не задан или равен dev-значению "
            f"({auth.jwt_secret!r}). Задайте случайные 64 символа."
        )
    elif len(auth.jwt_secret.strip()) < 32:
        issues.append("JWT_SECRET_KEY короче 32 символов: слишком слабый секрет.")

    if oauth.uses_default_cookie_secret:
        issues.append(
            "OAUTH_COOKIE_SECRET не задан: OAuth-куки подписываются JWT-секретом."
        )
    elif oauth.cookie_secret.strip() == auth.jwt_secret.strip():
        issues.append(
            "OAUTH_COOKIE_SECRET совпадает с JWT_SECRET_KEY: используйте "
            "отдельный секрет для OAuth-кук."
        )
    elif len(oauth.cookie_secret.strip()) < 32:
        issues.append(
            "OAUTH_COOKIE_SECRET короче 32 символов: слишком слабый секрет."
        )

    if not oauth.public_base_url.strip():
        issues.append(
            "OAUTH_PUBLIC_BASE_URL не задан: базовый адрес OAuth будет браться "
            "из заголовка Host (риск open redirect)."
        )
    elif not _is_https_url(oauth.public_base_url):
        issues.append(
            "OAUTH_PUBLIC_BASE_URL должен быть абсолютным https:// URL."
        )

    if not oauth.frontend_base_url.strip():
        issues.append(
            "FRONTEND_BASE_URL не задан: редирект после OAuth может уйти на "
            "подменённый Host."
        )
    elif not _is_https_url(oauth.frontend_base_url):
        issues.append("FRONTEND_BASE_URL должен быть абсолютным https:// URL.")

    if not bool(getattr(oauth, "cookie_secure", False)):
        issues.append(
            "OAUTH_COOKIE_SECURE=false: OAuth-кука должна передаваться только "
            "по HTTPS."
        )

    trusted_hosts = _as_list(security.trusted_hosts)
    if not trusted_hosts:
        issues.append(
            "TRUSTED_HOSTS не задан: список доверенных Host обязателен в production."
        )
    elif any(host.strip() == "*" for host in trusted_hosts):
        issues.append(
            "TRUSTED_HOSTS содержит '*': проверка Host фактически отключена."
        )

    origins = _as_list(cors.allowed_origins)
    if not origins:
        issues.append("ALLOWED_ORIGINS пуст: CORS не настроен.")
    else:
        if any(_is_local_origin(origin) for origin in origins):
            issues.append(
                "ALLOWED_ORIGINS содержит локальные адреса разработки: "
                "укажите домены production."
            )
        if any(origin.strip() == "*" for origin in origins):
            issues.append("ALLOWED_ORIGINS содержит '*': откройте только свои домены.")

    rate_limit = getattr(settings_obj, "rate_limit", None)
    if rate_limit is not None and not bool(getattr(rate_limit, "enabled", True)):
        issues.append(
            "RATE_LIMIT_ENABLED=false: в production лимиты частоты обязательны."
        )

    if bool(getattr(settings_obj, "debug", False)):
        issues.append("DEBUG=True: в production отладка должна быть выключена.")

    return issues


def validate_runtime_settings(settings_obj: Any) -> list[str]:
    """Проверяет конфигурацию. В production бросает SecurityConfigError.

    Возвращает список предупреждений для development-режима.
    """
    issues = collect_security_issues(settings_obj)
    if not issues:
        return []

    if getattr(settings_obj, "is_production", False):
        formatted = "\n".join(f"  - {issue}" for issue in issues)
        raise SecurityConfigError(
            "Небезопасная конфигурация для production:\n" + formatted
        )

    for issue in issues:
        logger.warning("[Security] %s", issue)
    return issues
