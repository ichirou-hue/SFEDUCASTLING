"""Точность ходов по этапам партии: дебют, миттельшпиль, эндшпиль.

Считается так же, как точность в разборе партии: оценка позиции
переводится в вероятность победы, а падение этой вероятности даёт
точность хода. Точность этапа - середина между средним арифметическим
и средним гармоническим по ходам этапа: одно грубое зедение не должно
растворяться в длинном списке спокойных ходов.

Ходы из уже решённых позиций не учитываются. Когда партия выиграна или
проиграна, вероятность победы упирается в предел, и любой ход выглядит
безупречным. Раньше из-за этого эндшпиль показывал 95-98 % даже в
партиях, проигранных ещё в дебюте.

Порог решённой позиции - 15 % вероятности победы. При пяти процентах на
живых партиях в миттельшпиле оставались только ходы из безнадёжных
позиций, и этап показывал 94 %, хотя всё было потеряно в дебюте: в
проигранной позиции почти любой ход сохраняет и без того низкие шансы.
"""

from __future__ import annotations

from typing import Iterable

import chess
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.analysis.win_probability import (
    DECIDED_WIN_PERCENT,
    classify,
    is_decided,
    move_accuracy,
    player_win_percent,
)
from backend.models.game import Game

PHASES = (
    ("opening", "Дебют"),
    ("middlegame", "Миттельшпиль"),
    ("endgame", "Эндшпиль"),
)
ERROR_CLASSES = ("inaccuracy", "mistake", "blunder")
QUALITY_CLASSES = ("best", "excellent", "good", *ERROR_CLASSES)
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
PLAYER_NAMES = {"player", "user"}

# Вероятность победы, за пределами которой ход уже ничего не решает.
# 15 %, а не 5 %: в безнадёжной позиции ход почти не меняет и без того
# низкие шансы, поэтому такой этап показывал бы 90 %+ даже в проигранной
# партии. Константа живёт в backend.analysis.win_probability: то же
# правило применяется в разборе партии.
# Меньше этого числа ходов - процент строить нельзя, он будет случайным.
MIN_SCORED_MOVES = 3

# Оценки в этом формате пишет game_recorder начиная с модели вероятности
# победы. Старые записи (сравнение с «лучшим ходом») не используем:
# их значения несопоставимы между собой.
EVAL_MODEL = "win_probability"


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


def _evaluation_pair(move) -> tuple[float, float] | None:
    """Пара (вероятность до, вероятность после) для хода игрока.

    ``None`` - оценок нет или они записаны старой моделью.
    """
    raw = move.eval_raw if isinstance(move.eval_raw, dict) else {}
    if raw.get("model") != EVAL_MODEL:
        return None
    if move.ev_before is None or move.ev_after is None:
        return None

    turn = chess.WHITE if move.turn == "w" else chess.BLACK
    return (
        player_win_percent(float(move.ev_before), turn),
        player_win_percent(float(move.ev_after), turn),
    )


def _harmonic_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return len(values) / sum(1.0 / max(value, 0.001) for value in values)


def _phase_accuracy(values: list[float]) -> float | None:
    """Середина арифметического и гармонического средних."""
    if len(values) < MIN_SCORED_MOVES:
        return None
    plain = sum(values) / len(values)
    return round((plain + _harmonic_mean(values)) / 2.0, 1)


def build_phase_accuracy(games: Iterable) -> dict:
    """Aggregate move accuracy and error counts for finished games."""
    counters = {
        phase: {
            "moves_total": 0,
            "moves_evaluated": 0,
            "moves_decided": 0,
            "moves_without_eval": 0,
            "accuracies": [],
            "quality": {name: 0 for name in QUALITY_CLASSES},
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

            pair = _evaluation_pair(move)
            if pair is None:
                row["moves_without_eval"] += 1
                continue

            win_before, win_after = pair
            row["moves_evaluated"] += 1

            if is_decided(win_before):
                row["moves_decided"] += 1
                continue

            classification = classify(win_before, win_after)
            row["quality"][classification] += 1
            row["accuracies"].append(move_accuracy(win_before, win_after))

    phases = []
    for phase, label in PHASES:
        row = counters[phase]
        phases.append({
            "key": phase,
            "label": label,
            "accuracy_percent": _phase_accuracy(row["accuracies"]),
            "moves_total": row["moves_total"],
            "moves_evaluated": row["moves_evaluated"],
            "moves_scored": len(row["accuracies"]),
            "moves_decided": row["moves_decided"],
            "moves_without_eval": row["moves_without_eval"],
            "errors": {
                name: row["quality"][name] for name in ERROR_CLASSES
            },
        })

    return {
        "games_analyzed": games_count,
        "phases": phases,
        "method": {
            "accuracy": (
                "Точность по падению вероятности победы (модель Lichess): "
                "середина среднего и гармонического среднего по ходам этапа. "
                "Ходы в решённых позициях (вероятность победы ниже 15 % или "
                "выше 85 %) не учитываются - в них качество хода ни о чём не "
                "говорит. Меньше трёх таких ходов на этап - процент не строится."
            ),
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
