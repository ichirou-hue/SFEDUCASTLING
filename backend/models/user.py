"""Модели пользователей и refresh-токенов.

`role` — единственный источник прав пользователя в БД: learner или admin.
Поле `is_admin` в API вычисляется из role только для совместимости со старым frontend.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


ROLE_LEARNER = "learner"
ROLE_ADMIN = "admin"
VALID_USER_ROLES = {ROLE_LEARNER, ROLE_ADMIN}


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    elo: Mapped[int | None] = mapped_column(Integer, nullable=True)
    role: Mapped[str] = mapped_column(
        String(32), default=ROLE_LEARNER, server_default=ROLE_LEARNER, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def effective_role(self) -> str:
        return self.role or ROLE_LEARNER

    def set_role(self, role: str) -> None:
        if role not in VALID_USER_ROLES:
            raise ValueError(f"Неизвестная роль: {role}")
        self.role = role

    def public(self) -> dict[str, Any]:
        """Данные пользователя для API без хеша пароля.

        `is_admin` оставлен только как вычисляемое поле ответа API,
        отдельной колонки users.is_admin в БД больше нет.
        """
        role = self.effective_role
        return {
            "id": self.id,
            "login": self.login,
            "email": self.email,
            "elo": self.elo,
            "role": role,
            "is_admin": role == ROLE_ADMIN,
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
