"""Permanent cosmetic items owned by a user."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, BigIntPK


class UserCosmetic(Base):
    __tablename__ = "user_cosmetics"
    __table_args__ = (
        UniqueConstraint("user_id", "cosmetic_id", name="uq_user_cosmetic"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cosmetic_id: Mapped[str] = mapped_column(String(64), nullable=False)
    is_equipped: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    purchased_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
