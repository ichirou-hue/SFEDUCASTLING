"""Модели партий и ходов (задача 81)."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    white: Mapped[str] = mapped_column(String(64), default="player")
    black: Mapped[str] = mapped_column(String(64), default="AI")
    result: Mapped[str] = mapped_column(String(8), default="*")
    date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    opening: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="playing")
    elo: Mapped[int | None] = mapped_column(Integer, nullable=True)
    engine: Mapped[str] = mapped_column(String(16), default="maia3")
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    moves: Mapped[list[GameMove]] = relationship(
        back_populates="game", cascade="all, delete-orphan", order_by="GameMove.move_no"
    )


class GameMove(Base):
    __tablename__ = "game_moves"
    __table_args__ = (UniqueConstraint("game_id", "move_no", name="uq_game_moves_game_move"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), index=True)
    move_no: Mapped[int] = mapped_column(Integer)
    fen: Mapped[str] = mapped_column(Text)
    uci: Mapped[str] = mapped_column(String(16))
    san: Mapped[str | None] = mapped_column(String(16), nullable=True)
    turn: Mapped[str] = mapped_column(String(8))
    classification: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ev_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    diff_cp: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_raw: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)

    game: Mapped[Game] = relationship(back_populates="moves")
