"""Уроки внутри учебного модуля."""

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class TrainingLesson(Base):
    __tablename__ = "training_lessons"
    __table_args__ = (
        UniqueConstraint("module_id", "slug", name="uq_training_lessons_module_slug"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    module_id: Mapped[int] = mapped_column(
        ForeignKey("training_modules.id", ondelete="CASCADE"),
        index=True,
    )
    slug: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(128))
    theory: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    module: Mapped["TrainingModule"] = relationship(back_populates="lessons")
    tasks: Mapped[list["TrainingTask"]] = relationship(
        back_populates="lesson",
        cascade="all, delete-orphan",
        order_by="TrainingTask.sort_order",
    )
