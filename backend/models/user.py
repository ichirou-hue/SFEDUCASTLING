"""Модели пользователей и refresh-токенов (задача 64).

Пароль в БД только в виде bcrypt-хеша (cost 12). Refresh-токены хранятся
как sha256-хеши — утечка БД не даёт угнать активные сессии.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base, JsonType


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    elo: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Сырые ответы онбординг-анкеты Q1-Q8. None — анкета ещё не заполнена
    # (по этому признаку фронтенд показывает плашку «Заполните анкету»).
    #
    # TODO(BACKEND): временный «мешок» для ответов. По методике нужны
    # отдельные колонки — prior_band (1..4), rating_estimate (шкала Lichess
    # blitz), rating_scale (напр. "lichess_blitz"), pedagogy (ответ Q6).
    # skill_band (0..4) пишет входной тест, не анкета.
    # Подробности — блок TODO в backend/api_gateway/routes/auth.py.
    onboarding: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def public(self) -> dict[str, Any]:
        """Данные пользователя для ответов API (без хеша пароля)."""
        return {
            "id": self.id,
            "login": self.login,
            "email": self.email,
            "elo": self.elo,
            "onboarding": self.onboarding,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    user: Mapped[User] = relationship(back_populates="refresh_tokens")
