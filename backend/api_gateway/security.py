"""Хеширование паролей и выпуск JWT (задача 64).

Пароли — bcrypt (cost 12). Access-токен — JWT HS256 с коротким TTL.
Refresh-токен — случайная строка; в БД хранится только её sha256-хеш.
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from backend.config.settings import settings

_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def create_access_token(user_id: int, login: str, is_admin: bool) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "login": login,
        "admin": is_admin,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.auth.access_token_ttl_min),
    }
    return jwt.encode(payload, settings.auth.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """Бросает jwt.PyJWTError, если токен невалиден/просрочен."""
    payload = jwt.decode(token, settings.auth.jwt_secret, algorithms=["HS256"])
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return payload


def generate_refresh_token() -> tuple[str, str]:
    """Возвращает (сырый токен для клиента, sha256-хеш для БД), срок из настроек."""
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def refresh_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(days=settings.auth.refresh_token_ttl_days)
