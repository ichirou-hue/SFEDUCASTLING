"""Дни, в которые пользователь заходил в приложение.

Нужны для серии дней (streak): серия считается по объединению дней
учебной активности (курс/паззлы) и дней визитов. Благодаря этому
счётчик начинается с 1 в день регистрации и растёт при ежедневных
заходах, даже если в этот день пользователь ничего не решал.
"""

from datetime import date, datetime

from sqlalchemy import Date, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, BigIntPK


class UserActivityDay(Base):
    __tablename__ = "user_activity_days"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "activity_date",
            name="uq_user_activity_days_user_date",
        ),
        Index("ix_user_activity_days_date", "activity_date"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    activity_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
