"""Сессия теста определения шахматного уровня.

Таблица нужна не только для истории, но и для идемпотентного завершения
level-test: повторный submit одного и того же теста возвращает уже сохранённый
результат и не пересчитывает/не перезаписывает профиль заново.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class LevelTest(Base):
    __tablename__ = "level_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # Фиксируем именно тот набор задач, который был выдан пользователю.
    # Это не даёт submit подменить тест произвольными puzzle_id.
    question_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)

    # Снимок финальных ответов и результата нужен для идемпотентного ответа.
    answers: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    score: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    band: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="started", server_default="started"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
