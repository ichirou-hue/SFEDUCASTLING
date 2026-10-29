"""Связанные шахматные аккаунты и аудит подтверждения владения."""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, JsonType

VERIFICATION_LINKED = "linked"
VERIFICATION_PENDING = "pending"
VERIFICATION_VERIFIED = "verified"
VERIFICATION_EXPIRED = "expired"
VERIFICATION_REVOKED = "revoked"
VERIFICATION_ERROR = "error"

VERIFICATION_METHOD_OAUTH = "oauth"
VERIFICATION_METHOD_ADMIN = "admin"
VERIFICATION_METHOD_USER_ACTION = "user_action"


class UserChessAccount(Base):
    __tablename__ = "user_chess_accounts"
    __table_args__ = (
        UniqueConstraint("user_id", "platform", name="uq_user_chess_accounts_user_platform"),
        UniqueConstraint(
            "platform",
            "external_user_id",
            name="uq_user_chess_accounts_platform_external_user_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    username: Mapped[str] = mapped_column(String(64), nullable=False)

    # Стабильный ID внешнего аккаунта появляется только после подтверждения владения.
    external_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    rating_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating_scale: Mapped[str | None] = mapped_column(String(64), nullable=True)
    games: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    rating_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    provisional: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    rating_usable: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)

    # verified оставлен для обратной совместимости с существующим кодом.
    verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    verification_status: Mapped[str] = mapped_column(
        String(32), default=VERIFICATION_LINKED, server_default=VERIFICATION_LINKED, nullable=False
    )
    verification_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verification_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verification_failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    profile_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    linked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    @property
    def ownership_verified(self) -> bool:
        return bool(self.verified and self.verification_status == VERIFICATION_VERIFIED)

    @property
    def rating_trusted(self) -> bool:
        """Можно ли использовать рейтинг как доверенный вход для assessment."""
        return bool(self.ownership_verified and self.rating_usable and self.rating is not None)


class ChessAccountVerification(Base):
    """Неизменяемый журнал попыток подтверждения владения внешним аккаунтом."""

    __tablename__ = "chess_account_verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chess_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("user_chess_accounts.id", ondelete="SET NULL"), index=True, nullable=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    expected_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    external_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    external_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
