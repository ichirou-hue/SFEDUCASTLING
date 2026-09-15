"""Текущая динамическая сложность пользователя по учебным темам."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class UserThemeDifficulty(Base):
    __tablename__ = "user_theme_difficulties"
    __table_args__ = (
        UniqueConstraint("user_id", "theme_slug", name="uq_user_theme_difficulty"),
        CheckConstraint(
            "current_difficulty >= 1 AND current_difficulty <= 3",
            name="ck_user_theme_difficulty_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    theme_slug: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    current_difficulty: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=2,
        server_default="2",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
