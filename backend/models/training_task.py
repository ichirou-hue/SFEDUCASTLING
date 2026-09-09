"""Стандартные учебные задания, отображаемые во вкладке «Обучение»."""

from typing import Any

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_id: Mapped[int] = mapped_column(
        ForeignKey("training_lessons.id", ondelete="CASCADE"),
        index=True,
    )

    # MVP: select_squares и make_move.
    # В дальнейшем сюда можно добавить choose_move, yes_no, find_capture и др.
    task_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(128))
    instruction: Mapped[str] = mapped_column(Text)

    # Исходная шахматная позиция задания.
    fen: Mapped[str] = mapped_column(Text)
    source_square: Mapped[str | None] = mapped_column(String(2), nullable=True)

    difficulty: Mapped[int] = mapped_column(Integer, default=1, server_default="1")

    # Дополнительные параметры конкретного типа задания.
    # Например: {"piece": "N", "mode": "all_legal_moves"}.
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Короткое объяснение, показываемое после проверки ответа.
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    sort_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    lesson: Mapped["TrainingLesson"] = relationship(back_populates="tasks")
    attempts: Mapped[list["TrainingAttempt"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="TrainingAttempt.created_at",
    )
