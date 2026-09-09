"""Учебные модули: верхний уровень курса (пешка, конь, слон и т.д.)."""

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class TrainingModule(Base):
    __tablename__ = "training_modules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    lessons: Mapped[list["TrainingLesson"]] = relationship(
        back_populates="module",
        cascade="all, delete-orphan",
        order_by="TrainingLesson.sort_order",
    )
