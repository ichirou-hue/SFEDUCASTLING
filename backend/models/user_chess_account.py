"""Связанные публичные шахматные аккаунты пользователя."""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class UserChessAccount(Base):
    __tablename__ = "user_chess_accounts"
    __table_args__ = (
        UniqueConstraint("user_id", "platform", name="uq_user_chess_accounts_user_platform"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    rating_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating_scale: Mapped[str | None] = mapped_column(String(64), nullable=True)
    games: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    rating_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    provisional: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    rating_usable: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    # До OAuth это всегда False: soft link подтверждает выбор профиля, не владение им.
    verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    profile_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    linked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
