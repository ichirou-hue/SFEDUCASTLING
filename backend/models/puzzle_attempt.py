"""Попытки решения тактических пазлов пользователями."""

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, BigIntPK


class PuzzleAttempt(Base):
    __tablename__ = "puzzle_attempts"
    __table_args__ = (
        Index("ix_puzzle_attempts_user_puzzle", "user_id", "puzzle_id"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    puzzle_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    correct: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
