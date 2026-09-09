"""Детерминированная проверка стандартных учебных заданий.

На первом этапе подсистема обучения не использует LLM и Stockfish:
ответы на базовые упражнения проверяются через python-chess.
"""

from dataclasses import dataclass
from typing import Any

import chess

from backend.models.training_task import TrainingTask


class TrainingCheckError(ValueError):
    """Некорректно настроенное задание или ответ неподходящего формата."""


@dataclass(slots=True)
class TrainingCheckResult:
    correct: bool
    score: float
    feedback: str
    expected: dict[str, Any]
    details: dict[str, Any]


def _board_for_task(task: TrainingTask) -> chess.Board:
    try:
        board = chess.Board(task.fen)
    except ValueError as exc:
        raise TrainingCheckError(f"Некорректный FEN в задании {task.id}: {exc}") from exc

    if not board.is_valid():
        raise TrainingCheckError(f"Позиция задания {task.id} не является корректной")
    return board


def _source_square(task: TrainingTask) -> int:
    if not task.source_square:
        raise TrainingCheckError("Для задания не задано source_square")
    try:
        return chess.parse_square(task.source_square.lower())
    except ValueError as exc:
        raise TrainingCheckError("Некорректное source_square") from exc


def _legal_moves_from(board: chess.Board, source_square: int) -> list[chess.Move]:
    return [move for move in board.legal_moves if move.from_square == source_square]


def _mode(task: TrainingTask) -> str:
    payload = task.payload or {}
    return str(payload.get("mode") or "all_legal_moves")


def _moves_for_mode(
    board: chess.Board,
    task: TrainingTask,
    source_square: int,
) -> list[chess.Move]:
    legal = _legal_moves_from(board, source_square)
    mode = _mode(task)

    if mode in {"all_legal_moves", "any_legal_move"}:
        return legal
    if mode in {"capture_squares", "legal_capture"}:
        return [move for move in legal if board.is_capture(move)]
    if mode == "accepted_moves":
        accepted = set((task.payload or {}).get("accepted_moves") or [])
        return [move for move in legal if move.uci() in accepted]

    raise TrainingCheckError(f"Неизвестный режим проверки: {mode}")


def _normalize_squares(value: Any) -> set[str]:
    if not isinstance(value, list):
        raise TrainingCheckError("Ожидается массив selected_squares")

    result: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TrainingCheckError("Названия клеток должны быть строками")
        square = item.strip().lower()
        try:
            chess.parse_square(square)
        except ValueError as exc:
            raise TrainingCheckError(f"Некорректная клетка: {item}") from exc
        result.add(square)
    return result


def _check_select_squares(
    board: chess.Board,
    task: TrainingTask,
    answer: dict[str, Any],
) -> TrainingCheckResult:
    source = _source_square(task)
    expected_moves = _moves_for_mode(board, task, source)
    expected = {chess.square_name(move.to_square) for move in expected_moves}
    selected = _normalize_squares(answer.get("selected_squares"))

    correct = selected == expected
    union = selected | expected
    score = 1.0 if not union else len(selected & expected) / len(union)

    missing = sorted(expected - selected)
    extra = sorted(selected - expected)

    if correct:
        feedback = "Верно. Все необходимые клетки отмечены правильно."
    elif missing and extra:
        feedback = "Ответ частично верный: есть пропущенные и лишние клетки."
    elif missing:
        feedback = "Ответ частично верный: отмечены не все доступные клетки."
    else:
        feedback = "В ответе отмечены клетки, которые не подходят к условию."

    return TrainingCheckResult(
        correct=correct,
        score=round(score, 3),
        feedback=feedback,
        expected={"selected_squares": sorted(expected)},
        details={"missing": missing, "extra": extra},
    )


def _check_make_move(
    board: chess.Board,
    task: TrainingTask,
    answer: dict[str, Any],
) -> TrainingCheckResult:
    raw_move = answer.get("move")
    if not isinstance(raw_move, str) or not raw_move.strip():
        raise TrainingCheckError("Ожидается поле move в формате UCI")

    try:
        move = chess.Move.from_uci(raw_move.strip().lower())
    except ValueError as exc:
        raise TrainingCheckError("Некорректный UCI-ход") from exc

    source = _source_square(task)
    acceptable = _moves_for_mode(board, task, source)
    acceptable_uci = sorted(m.uci() for m in acceptable)

    from_expected_square = move.from_square == source
    legal = move in board.legal_moves
    correct = from_expected_square and move in acceptable

    mode = _mode(task)
    if correct:
        feedback = "Верно. Ход соответствует условию задания."
    elif not from_expected_square:
        feedback = f"Нужно сделать ход фигурой с поля {task.source_square}."
    elif not legal:
        feedback = "Этот ход нелегален в данной позиции."
    elif mode in {"capture_squares", "legal_capture"}:
        feedback = "Ход легален, но по условию необходимо выполнить взятие."
    else:
        feedback = "Ход легален, но не соответствует условию задания."

    return TrainingCheckResult(
        correct=correct,
        score=1.0 if correct else 0.0,
        feedback=feedback,
        expected={"moves": acceptable_uci},
        details={"legal": legal, "from_expected_square": from_expected_square},
    )


def check_training_task(
    task: TrainingTask,
    answer: dict[str, Any],
) -> TrainingCheckResult:
    """Проверяет ответ в зависимости от типа стандартного задания."""
    if not isinstance(answer, dict):
        raise TrainingCheckError("answer должен быть JSON-объектом")

    board = _board_for_task(task)

    if task.task_type == "select_squares":
        return _check_select_squares(board, task, answer)
    if task.task_type == "make_move":
        return _check_make_move(board, task, answer)

    raise TrainingCheckError(f"Тип задания {task.task_type!r} пока не поддерживается")
