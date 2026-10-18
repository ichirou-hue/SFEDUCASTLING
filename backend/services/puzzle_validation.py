"""Единая проверка качества тактических пазлов и учебных заданий курса.

Модуль не зависит от БД и FastAPI: только python-chess. Используется
сборщиками контента и тестами, чтобы гарантировать, что:

- FEN корректна, а ходы решения легальны;
- после решения игрок не матован и не остаётся под шахом;
- «целевая» фигура темы (конь, ладья, пешка и т.д.) реально участвует
  в решении пазла;
- в учебном задании целевая фигура стоит на объявленном поле.
"""

from __future__ import annotations

from typing import Any

import chess

# Lichess-темы, которые называют конкретную «целевую» фигуру, и её символ.
PIECE_THEMES: dict[str, str] = {
    "pawnEndgame": "P",
    "advancedPawn": "P",
    "promotion": "P",
    "underPromotion": "P",
    "knightEndgame": "N",
    "bishopEndgame": "B",
    "rookEndgame": "R",
    "queenEndgame": "Q",
}

PIECE_TYPE_BY_SYMBOL: dict[str, int] = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}

# Типы заданий курса и режимы проверки (синхронизированы с training_checker).
SELECT_TASK_TYPES = {"select_squares", "make_move"}
VALID_TASK_TYPES = {"select_squares", "make_move", "choose_option"}
VALID_MODES = {
    "all_legal_moves",
    "any_legal_move",
    "capture_squares",
    "legal_capture",
    "accepted_moves",
}


def board_from_fen(fen: str | None) -> chess.Board | None:
    """Возвращает доску по FEN или None, если FEN не разбирается."""
    if not fen:
        return None
    try:
        return chess.Board(fen)
    except ValueError:
        return None


def parse_uci_moves(moves: Any) -> list[chess.Move] | None:
    """Разбирает строку UCI-ходов (например, "c4c5 g4h4 c7c8q") в список ходов.

    Ходы разделены пробелами. Превращение записывается пятью символами
    (c7c8q), поэтому длину жёстко не фиксируем.
    """
    text = str(moves or "").strip()
    if not text:
        return None
    parsed: list[chess.Move] = []
    for token in text.split():
        if len(token) not in (4, 5):
            return None
        try:
            parsed.append(chess.Move.from_uci(token))
        except ValueError:
            return None
    return parsed


def pieces_used_in_solution(fen: str | None, moves: Any) -> set[int] | None:
    """Типы фигур, которые делают ход в решении.

    Возвращает None, если позицию или ходы разобрать нельзя.
    """
    board = board_from_fen(fen)
    parsed = parse_uci_moves(moves)
    if board is None or not parsed:
        return None

    used: set[int] = set()
    for move in parsed:
        piece = board.piece_at(move.from_square)
        if piece is None:
            return None
        try:
            board.push(move)
        except Exception:
            return None
        used.add(piece.piece_type)
    return used


def pieces_on_board(fen: str | None) -> set[int] | None:
    """Типы фигур, присутствующие на доске (цвет не важен)."""
    board = board_from_fen(fen)
    if board is None:
        return None
    return {piece.piece_type for piece in board.piece_map().values()}


def target_piece_used_in_solution(puzzle: dict, symbol: str) -> bool | None:
    """Участвует ли фигура symbol в решении. None — проверить нельзя."""
    piece_type = PIECE_TYPE_BY_SYMBOL.get(symbol.upper())
    if piece_type is None:
        return None
    used = pieces_used_in_solution(puzzle.get("fen"), puzzle.get("moves"))
    if used is None:
        return None
    return piece_type in used


def puzzle_themes_piece_ok(puzzle: dict) -> bool:
    """Все ли целевые фигуры названных тем участвуют в решении пазла."""
    themes = set(puzzle.get("themes") or [])
    wanted = {PIECE_THEMES[theme] for theme in themes if theme in PIECE_THEMES}
    if not wanted:
        return True
    used = pieces_used_in_solution(puzzle.get("fen"), puzzle.get("moves"))
    if used is None:
        # Позицию разобрать нельзя — откатываемся к наличию фигуры на доске,
        # чтобы не отбраковывать данные из-за парсинга.
        on_board = pieces_on_board(puzzle.get("fen"))
        if on_board is None:
            return True
        return all(PIECE_TYPE_BY_SYMBOL[symbol] in on_board for symbol in wanted)
    return all(PIECE_TYPE_BY_SYMBOL[symbol] in used for symbol in wanted)


def validate_puzzle(puzzle: dict, check_piece: bool = True) -> list[str]:
    """Возвращает список ошибок пазла (пусто — данные корректны).

    check_piece=False отключает проверку участия целевой фигуры: она строже
    требований Lichess и подходит не всем пазлам (например, тема мата может
    называть фигуру, которой в решении нет). Для отбора контента её включают.
    """
    errors: list[str] = []

    if not puzzle.get("id"):
        errors.append("нет id")

    fen = puzzle.get("fen")
    board = board_from_fen(fen)
    if board is None:
        errors.append("некорректный FEN")
        return errors

    parsed = parse_uci_moves(puzzle.get("moves"))
    if not parsed:
        errors.append("некорректная последовательность ходов")
        return errors

    for move in parsed:
        if move not in board.legal_moves:
            errors.append(f"нелегальный ход {move.uci()}")
            return errors
        board.push(move)

    initial_turn = str(fen).split()[1] if fen else "w"
    player_color = chess.WHITE if initial_turn == "w" else chess.BLACK
    if board.is_checkmate() and board.turn == player_color:
        errors.append("игрок получает мат")
    if board.is_check() and board.turn == player_color:
        errors.append("игрок остаётся под шахом")

    if check_piece and not puzzle_themes_piece_ok(puzzle):
        errors.append("целевая фигура темы не участвует в решении")

    return errors


def validate_training_task(task: dict) -> list[str]:
    """Возвращает список ошибок учебного задания (пусто — корректно)."""
    errors: list[str] = []

    task_type = task.get("task_type")
    if task_type not in VALID_TASK_TYPES:
        errors.append(f"неизвестный тип задания {task_type!r}")

    board = board_from_fen(task.get("fen"))
    if board is None or not board.is_valid():
        errors.append("некорректный FEN")
        return errors

    payload = task.get("payload") or {}
    if not isinstance(payload, dict):
        errors.append("payload должен быть объектом")
        return errors

    if task_type == "choose_option":
        options = payload.get("options")
        correct_option = payload.get("correct_option")
        if not isinstance(options, list) or not options:
            errors.append("choose_option без options")
        else:
            ids = {
                option.get("id")
                for option in options
                if isinstance(option, dict)
            }
            if correct_option not in ids:
                errors.append("correct_option отсутствует в options")
        return errors

    if task_type in SELECT_TASK_TYPES:
        source = task.get("source_square")
        if not source:
            errors.append("не задано source_square")
            return errors
        try:
            source_square = chess.parse_square(str(source).lower())
        except ValueError:
            errors.append("некорректное source_square")
            return errors

        piece = board.piece_at(source_square)
        if piece is None:
            errors.append(f"на поле {source} нет фигуры")
            return errors

        expected_symbol = payload.get("piece")
        if expected_symbol and piece.symbol().upper() != str(expected_symbol).upper():
            errors.append(
                f"на {source} стоит {piece.symbol().upper()}, "
                f"а в payload.piece указана {expected_symbol}"
            )

        mode = str(payload.get("mode") or "all_legal_moves")
        if mode not in VALID_MODES:
            errors.append(f"неизвестный режим {mode!r}")
            return errors

        legal = [m for m in board.legal_moves if m.from_square == source_square]
        if mode in {"capture_squares", "legal_capture"}:
            legal = [m for m in legal if board.is_capture(m)]
        if mode == "accepted_moves":
            accepted = {
                str(m).strip().lower()
                for m in (payload.get("accepted_moves") or [])
            }
            if not accepted:
                errors.append("accepted_moves пуст")
            legal = [m for m in legal if m.uci() in accepted]
        if not legal:
            errors.append("нет ни одного допустимого хода")

    return errors
