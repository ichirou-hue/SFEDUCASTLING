"""Сессия теста определения шахматного уровня.

Таблица нужна не только для истории, но и для идемпотентного завершения
level-test: повторный submit одного и того же теста возвращает уже сохранённый
результат и не пересчитывает/не перезаписывает профиль заново.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, JsonType


class LevelTest(Base):
    __tablename__ = "level_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # Фиксируем именно тот набор задач, который был выдан пользователю.
    # Это не даёт submit подменить тест произвольными puzzle_id.
    question_ids: Mapped[list[str]] = mapped_column(JsonType, nullable=False)

    # Персонализация конкретной попытки. seed позволяет воспроизвести набор,
    # initial_rating и rating_group фиксируют стартовую гипотезу, а
    # metrics_snapshot объясняет, какие метрики повлияли на выборку.
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    initial_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating_group: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metrics_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)

    # answers хранит и промежуточные ответы started-теста, и финальный снимок.
    answers: Mapped[list[dict[str, Any]] | None] = mapped_column(JsonType, nullable=True)
    score: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)

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
