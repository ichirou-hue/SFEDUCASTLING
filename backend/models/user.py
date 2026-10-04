"""Модели пользователей и refresh-токенов (задача 64).

Пароль в БД только в виде bcrypt-хеша (cost 12). Refresh-токены хранятся
как sha256-хеши — утечка БД не даёт угнать активные сессии.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    elo: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Роли: guest (без аккаунта, в БД нет строки) / learner (по умолчанию) / admin.
    # `role` — единственный источник истины; is_admin рассчитывается от него.
    role: Mapped[str] = mapped_column(
        String(16), default="learner", server_default="learner", nullable=False
    )

    # Методичка оценки уровня (см. docs): онбординг → prior_band,
    # внешний рейтинг → rating_estimate/rating_scale, входной тест → skill_band (0–4).
    skill_band: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prior_band: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating_scale: Mapped[str | None] = mapped_column(String(32), nullable=True)
    onboarding: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    parental_consent: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    @property
    def is_admin(self) -> bool:
        """Совместимость: администратор = роль admin (колонки is_admin больше нет)."""
        return self.role == "admin"

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
            "is_admin": self.is_admin,
            "role": self.role,
            "skill_band": self.skill_band,
            "prior_band": self.prior_band,
            "rating_estimate": self.rating_estimate,
            "rating_scale": self.rating_scale,
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
