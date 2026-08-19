"""Модель сообщения чата (задача 65)."""

from datetime import UTC, datetime

from sqlalchemy import BigInteger, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


def _now_ts() -> float:
    return datetime.now(UTC).timestamp()


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(16), default="assistant")
    text: Mapped[str] = mapped_column(Text)
    ts: Mapped[float] = mapped_column(default=_now_ts, index=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
