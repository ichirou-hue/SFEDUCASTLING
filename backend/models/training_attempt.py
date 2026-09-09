"""Попытки решения учебных заданий пользователями."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class TrainingAttempt(Base):
    __tablename__ = "training_attempts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    task_id: Mapped[int] = mapped_column(
        ForeignKey("training_tasks.id", ondelete="CASCADE"),
        index=True,
    )

    # Универсальный JSON-ответ:
    # select_squares -> {"selected_squares": ["c3", "d2", ...]}
    # make_move      -> {"move": "e4f6"}
    answer: Mapped[dict[str, Any]] = mapped_column(JSONB)

    correct: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    score: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    hints_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    task: Mapped["TrainingTask"] = relationship(back_populates="attempts")
