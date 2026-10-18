"""Aggregate a user's move quality by opening, middlegame, and endgame."""

from __future__ import annotations

from typing import Iterable

import chess
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models.game import Game


PHASES = (
    ("opening", "Дебют"),
    ("middlegame", "Миттельшпиль"),
    ("endgame", "Эндшпиль"),
)
GOOD_CLASSES = {"best", "excellent", "good"}
ERROR_CLASSES = ("inaccuracy", "mistake", "blunder")
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
PLAYER_NAMES = {"player", "user"}


def _player_color(game) -> chess.Color:
    white = str(game.white or "").casefold()
    black = str(game.black or "").casefold()
    if black in PLAYER_NAMES and white not in PLAYER_NAMES:
        return chess.BLACK
    return chess.WHITE


def _phase_for_position(fen: str, move_no: int) -> str:
    """Use first ten full moves for opening, then a material-based endgame test."""
    board = chess.Board(fen)
    fullmove_number = (move_no + 1) // 2
    if fullmove_number <= 10:
        return "opening"

    queens_present = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
    non_pawn_material = sum(
        len(board.pieces(piece_type, color)) * value
        for color in (chess.WHITE, chess.BLACK)
        for piece_type, value in (
            (chess.KNIGHT, 3),
            (chess.BISHOP, 3),
            (chess.ROOK, 5),
        )
    )
    if not queens_present or non_pawn_material <= 13:
        return "endgame"
    return "middlegame"


def build_phase_accuracy(games: Iterable) -> dict:
    """Return evaluated move quality and error counts for the user's finished games.

    Accuracy is the share of evaluated user moves classified as best, excellent,
    or good. Moves without a Stockfish classification are reported separately.
    """
    counters = {
        phase: {
            "moves_total": 0,
            "moves_evaluated": 0,
            "moves_without_eval": 0,
            "quality": {name: 0 for name in ("best", "excellent", "good", *ERROR_CLASSES)},
        }
        for phase, _ in PHASES
    }
    games_count = 0

    for game in games:
        if game.result not in VALID_RESULTS:
            continue
        games_count += 1
        player_color = _player_color(game)
        for move in game.moves:
            is_player_move = (move.turn == "w") == (player_color == chess.WHITE)
            if not is_player_move:
                continue

            phase = _phase_for_position(move.fen, move.move_no)
            row = counters[phase]
            row["moves_total"] += 1
            classification = move.classification
            if classification not in row["quality"]:
                row["moves_without_eval"] += 1
                continue
            row["moves_evaluated"] += 1
            row["quality"][classification] += 1

    phases = []
    for phase, label in PHASES:
        row = counters[phase]
        evaluated = row["moves_evaluated"]
        good_moves = sum(row["quality"][name] for name in GOOD_CLASSES)
        phases.append({
            "key": phase,
            "label": label,
            "accuracy_percent": round(good_moves * 100 / evaluated, 1) if evaluated else None,
            "moves_total": row["moves_total"],
            "moves_evaluated": evaluated,
            "moves_without_eval": row["moves_without_eval"],
            "errors": {
                name: row["quality"][name] for name in ERROR_CLASSES
            },
        })

    return {
        "games_analyzed": games_count,
        "phases": phases,
        "method": {
            "accuracy": "Доля ходов best, excellent и good среди ходов с оценкой Stockfish.",
            "opening_full_moves": 10,
            "endgame": "После 10-го хода, если ферзи сняты или осталось не более 13 очков фигур (без пешек и королей).",
        },
    }


async def load_user_phase_accuracy(db: AsyncSession, *, user_id: int) -> dict:
    """Load completed games with moves and aggregate their phase accuracy."""
    games = await db.scalars(
        select(Game)
        .options(selectinload(Game.moves))
        .where(Game.user_id == user_id, Game.result.in_(VALID_RESULTS))
        .order_by(Game.created_at, Game.id)
    )
    return build_phase_accuracy(games.all())
