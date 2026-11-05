"""Weak/strong signals extracted from completed games against Maia."""

from __future__ import annotations

from typing import Iterable

import chess
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.analysis.win_probability import classify, is_decided, move_accuracy, player_win_percent
from backend.models.game import Game
from backend.services.phase_accuracy import build_phase_accuracy

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
PLAYER_NAMES = {"player", "user"}
EVAL_MODEL = "win_probability"
MIN_PIECE_SCORED_MOVES = 5
QUALITY_CLASSES = ("best", "excellent", "good", "inaccuracy", "mistake", "blunder")
ERROR_CLASSES = ("inaccuracy", "mistake", "blunder")
ERROR_WEIGHTS = {"inaccuracy": 1, "mistake": 2, "blunder": 4}

PIECES = (
    ("pawn", "Пешка", chess.PAWN),
    ("knight", "Конь", chess.KNIGHT),
    ("bishop", "Слон", chess.BISHOP),
    ("rook", "Ладья", chess.ROOK),
    ("queen", "Ферзь", chess.QUEEN),
    ("king", "Король", chess.KING),
)
PIECE_TYPE_TO_KEY = {piece_type: key for key, _label, piece_type in PIECES}


def _player_color(game) -> chess.Color:
    white = str(getattr(game, "white", "") or "").casefold()
    black = str(getattr(game, "black", "") or "").casefold()
    if black in PLAYER_NAMES and white not in PLAYER_NAMES:
        return chess.BLACK
    return chess.WHITE


def _evaluation_pair(move) -> tuple[float, float] | None:
    raw = move.eval_raw if isinstance(getattr(move, "eval_raw", None), dict) else {}
    if raw.get("model") != EVAL_MODEL:
        return None
    if getattr(move, "ev_before", None) is None or getattr(move, "ev_after", None) is None:
        return None
    turn = chess.WHITE if move.turn == "w" else chess.BLACK
    return (
        player_win_percent(float(move.ev_before), turn),
        player_win_percent(float(move.ev_after), turn),
    )


def _piece_key(move) -> str | None:
    try:
        board = chess.Board(move.fen)
        uci = chess.Move.from_uci(move.uci)
    except (ValueError, TypeError):
        return None
    piece = board.piece_at(uci.from_square)
    return PIECE_TYPE_TO_KEY.get(piece.piece_type) if piece else None


def _harmonic_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return len(values) / sum(1.0 / max(value, 0.001) for value in values)


def _accuracy(values: list[float]) -> float | None:
    if len(values) < MIN_PIECE_SCORED_MOVES:
        return None
    plain = sum(values) / len(values)
    return round((plain + _harmonic_mean(values)) / 2.0, 1)


def build_game_insights(games: Iterable) -> dict:
    """Aggregate Maia games into piece, phase and move-quality weaknesses."""
    game_list = [
        game for game in games
        if getattr(game, "result", None) in VALID_RESULTS
        and str(getattr(game, "engine", "maia3") or "maia3").casefold() == "maia3"
    ]
    counters = {
        key: {
            "moves_total": 0,
            "moves_evaluated": 0,
            "moves_decided": 0,
            "moves_without_eval": 0,
            "accuracies": [],
            "quality": {name: 0 for name in QUALITY_CLASSES},
        }
        for key, _label, _piece_type in PIECES
    }
    quality = {name: 0 for name in QUALITY_CLASSES}
    brilliant_moves = 0

    for game in game_list:
        player_color = _player_color(game)
        for move in getattr(game, "moves", []) or []:
            is_player_move = (move.turn == "w") == (player_color == chess.WHITE)
            if not is_player_move:
                continue

            raw = move.eval_raw if isinstance(getattr(move, "eval_raw", None), dict) else {}
            if raw.get("brilliant") is True:
                brilliant_moves += 1

            key = _piece_key(move)
            if key is None:
                continue
            row = counters[key]
            row["moves_total"] += 1

            pair = _evaluation_pair(move)
            if pair is None:
                row["moves_without_eval"] += 1
                continue
            row["moves_evaluated"] += 1
            win_before, win_after = pair
            if is_decided(win_before):
                row["moves_decided"] += 1
                continue

            cls = classify(win_before, win_after)
            accuracy = move_accuracy(win_before, win_after)
            row["quality"][cls] += 1
            row["accuracies"].append(accuracy)
            quality[cls] += 1

    pieces = []
    for key, label, _piece_type in PIECES:
        row = counters[key]
        accuracy_percent = _accuracy(row["accuracies"])
        error_score = sum(row["quality"][name] * ERROR_WEIGHTS[name] for name in ERROR_CLASSES)
        pieces.append({
            "key": key,
            "label": label,
            "accuracy_percent": accuracy_percent,
            "moves_total": row["moves_total"],
            "moves_evaluated": row["moves_evaluated"],
            "moves_scored": len(row["accuracies"]),
            "moves_decided": row["moves_decided"],
            "moves_without_eval": row["moves_without_eval"],
            "errors": {name: row["quality"][name] for name in ERROR_CLASSES},
            "quality": dict(row["quality"]),
            "error_score": error_score,
            "observed": len(row["accuracies"]) >= MIN_PIECE_SCORED_MOVES,
        })

    reliable = [piece for piece in pieces if piece["observed"] and piece["accuracy_percent"] is not None]
    weak_pieces = sorted(
        reliable,
        key=lambda item: (float(item["accuracy_percent"]), -int(item["error_score"]), item["key"]),
    )[:3]
    weak_keys = {item["key"] for item in weak_pieces}
    strong_pieces = sorted(
        (item for item in reliable if item["key"] not in weak_keys),
        key=lambda item: (-float(item["accuracy_percent"]), int(item["error_score"]), item["key"]),
    )[:3]

    phase_profile = build_phase_accuracy(game_list)
    errors_total = sum(quality[name] for name in ERROR_CLASSES)
    return {
        "games_analyzed": len(game_list),
        "pieces": pieces,
        "weak_pieces": weak_pieces,
        "strong_pieces": strong_pieces,
        "phases": phase_profile["phases"],
        "quality": {
            **quality,
            "errors_total": errors_total,
            "brilliant_moves": brilliant_moves,
        },
        "method": {
            "piece_min_scored_moves": MIN_PIECE_SCORED_MOVES,
            "error_weights": dict(ERROR_WEIGHTS),
            "source": "completed_maia_games",
        },
    }


async def load_user_game_insights(db: AsyncSession, *, user_id: int) -> dict:
    games = await db.scalars(
        select(Game)
        .options(selectinload(Game.moves))
        .where(
            Game.user_id == user_id,
            Game.result.in_(VALID_RESULTS),
            Game.engine == "maia3",
        )
        .order_by(Game.created_at, Game.id)
    )
    return build_game_insights(games.all())
