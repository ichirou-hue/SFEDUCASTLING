"""Модель хода в тренировочном датасете (задача 67)."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class DatasetMove(Base):
    __tablename__ = "dataset_moves"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fen: Mapped[str] = mapped_column(Text)
    user_move: Mapped[str] = mapped_column(String(16))
    stockfish_move: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stockfish_eval: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    user_id: Mapped[str] = mapped_column(String(64), default="anonymous", index=True)
    game_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
