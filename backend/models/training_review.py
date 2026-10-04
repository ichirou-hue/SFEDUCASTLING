"""Интервальные повторения учебных заданий пользователя (B3)."""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class TrainingReview(Base):
    __tablename__ = "training_reviews"
    __table_args__ = (
        UniqueConstraint("user_id", "task_id", name="uq_training_review_user_task"),
        CheckConstraint("ease >= 1.0", name="ck_training_review_ease"),
        CheckConstraint("reps >= 0", name="ck_training_review_reps"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id: Mapped[int] = mapped_column(
        ForeignKey("training_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    next_review_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    # Lite-вариант SM-2: пока коэффициент постоянный 2.2, но поле оставляем
    # на уровне пользователя/задания для будущей персонализации алгоритма.
    ease: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=2.2,
        server_default="2.2",
    )
    # Количество последовательных успешных повторений после последней ошибки.
    reps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
