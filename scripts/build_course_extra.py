"""Сборка дополнительных заданий курса из реальных партий.

Источники:
- ``backend/knowledge/puzzles.json`` - задачи Lichess (CC0) с темами;
- ``backend/knowledge/puzzle_extra.json`` - добор редких тем (тоже Lichess);
- ``backend/knowledge/openings.json`` - теория дебютов (lichess-org/chess-openings).

Оба файла задач хранятся в формате проекта: FEN - позиция, где ход у
решающего (белые), ``moves`` - линия начиная с хода белых. Записи Lichess
приводятся к нему скриптом ``scripts/build_puzzle_extra.py``.

Скрипт детерминированный: при одинаковых входных файлах результат совпадает.
Запуск из корня проекта::

    python scripts/build_course_extra.py

Результат пишется в ``backend/services/course_extra_tasks.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
PUZZLES = ROOT / "backend" / "knowledge" / "puzzles.json"
EXTRA = ROOT / "backend" / "knowledge" / "puzzle_extra.json"
OUT = ROOT / "backend" / "services" / "course_extra_tasks.py"

VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
PIECE_RU = {
    chess.PAWN: "пешка",
    chess.KNIGHT: "конь",
    chess.BISHOP: "слон",
    chess.ROOK: "ладья",
    chess.QUEEN: "ферзь",
    chess.KING: "король",
}
PIECE_GEN = {
    chess.PAWN: "пешки",
    chess.KNIGHT: "коня",
    chess.BISHOP: "слона",
    chess.ROOK: "ладьи",
    chess.QUEEN: "ферзя",
    chess.KING: "короля",
}
PIECE_ACC = {
    chess.PAWN: "пешку",
    chess.KNIGHT: "коня",
    chess.BISHOP: "слона",
    chess.ROOK: "ладью",
    chess.QUEEN: "ферзя",
    chess.KING: "короля",
}
PIECE_INSTR = {
    chess.PAWN: "пешкой",
    chess.KNIGHT: "конём",
    chess.BISHOP: "слоном",
    chess.ROOK: "ладьёй",
    chess.QUEEN: "ферзём",
    chess.KING: "королём",
}
PIECE_ADJ = {
    chess.PAWN: "белая",
    chess.KNIGHT: "белый",
    chess.BISHOP: "белый",
    chess.ROOK: "белая",
    chess.QUEEN: "белый",
    chess.KING: "белый",
}


def _piece_phrase(square_name: str, piece_type: int) -> str:
    return f"{PIECE_ADJ[piece_type]} {PIECE_RU[piece_type]} {square_name}"


def load_pool() -> list[dict]:
    data = json.loads(PUZZLES.read_text(encoding="utf-8"))
    pool = list(data["puzzles"])
    if EXTRA.exists():
        extra = json.loads(EXTRA.read_text(encoding="utf-8"))
        pool.extend(extra["puzzles"])
    seen: set[str] = set()
    result = []
    for item in pool:
        key = item.get("id") or item.get("fen")
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def material(board: chess.Board, color: bool) -> int:
    return sum(
        VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == color
    )


def solution(fen: str, moves: str) -> list[chess.Move] | None:
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    sequence = "".join(str(moves).split())
    result = []
    for index in range(0, len(sequence) - 3, 4):
        try:
            move = chess.Move.from_uci(sequence[index : index + 4])
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        result.append(move)
        board.push(move)
    return result or None


def san_line(board: chess.Board, moves: list[chess.Move], limit: int = 6) -> str:
    parts: list[str] = []
    probe = board.copy()
    for move in moves[:limit]:
        prefix = f"{probe.fullmove_number}." if probe.turn == chess.WHITE else f"{probe.fullmove_number}..."
        parts.append(prefix + probe.san(move))
        probe.push(move)
    if len(moves) > limit:
        parts.append("...")
    return " ".join(parts)


def facts(puzzle: dict) -> dict | None:
    board = chess.Board(puzzle["fen"])
    if board.turn != chess.WHITE:
        return None
    moves = solution(puzzle["fen"], puzzle["moves"])
    if not moves:
        return None
    first = moves[0]
    piece = board.piece_at(first.from_square)
    if piece is None:
        return None
    captured = board.piece_at(first.to_square)
    start_material = material(board, chess.WHITE) - material(board, chess.BLACK)

    san_first = board.san(first)
    after = board.copy()
    after.push(first)
    check = after.is_check()
    mate = after.is_checkmate()

    final = board.copy()
    for move in moves:
        final.push(move)
    delta = (material(final, chess.WHITE) - material(final, chess.BLACK)) - start_material

    # Фигуры соперника, которые атакует сделавшая ход фигура.
    attacked = []
    for square in after.attacks(first.to_square):
        target = after.piece_at(square)
        if target is not None and target.color == chess.BLACK:
            attacked.append((target.piece_type, square))

    discovered_check = False
    if check and not mate:
        king_square = after.king(chess.BLACK)
        discovered_check = king_square is not None and king_square not in after.attacks(
            first.to_square
        )

    return {
        "board": board,
        "moves": moves,
        "first": first,
        "piece": piece,
        "captured": captured,
        "san_first": san_first,
        "check": check,
        "mate": mate,
        "final": final,
        "final_mate": final.is_checkmate(),
        "solver_mated": final.is_checkmate() and final.turn == chess.WHITE,
        "delta": delta,
        "attacked": attacked,
        "discovered_check": discovered_check,
        "quiet": not board.is_capture(first) and not check,
    }


def _rating_band(rating: int, low: int, high: int) -> bool:
    return low <= rating <= high


def pick(pool, spec, used_fens, used_ids):
    low, high = spec.get("rating", (500, 2000))
    themes = set(spec.get("themes", ()))
    target = spec.get("target", (low + high) // 2)
    require = spec.get("require")
    exclusions = set(spec.get("exclude_themes", ()))

    candidates = []
    for puzzle in pool:
        pid = puzzle.get("id")
        fen = puzzle.get("fen")
        if pid in used_ids or fen in used_fens:
            continue
        rating = int(puzzle.get("rating") or 0)
        if not _rating_band(rating, low, high):
            continue
        puzzle_themes = set(puzzle.get("themes") or ())
        if themes and not (puzzle_themes & themes):
            continue
        if exclusions & puzzle_themes:
            continue
        info = facts(puzzle)
        if info is None:
            continue
        # Задача из партии должна быть выигрышной для белых: если линия
        # заканчивается матом белому королю, это позиция проигравшей стороны.
        if info["solver_mated"]:
            continue
        if require is not None and not require(info, puzzle_themes):
            continue
        candidates.append((abs(rating - target), rating, pid or "", puzzle, info))

    candidates.sort(key=lambda item: (item[0], item[2]))
    selected = []
    for _, _, pid, puzzle, info in candidates:
        if len(selected) >= spec["count"]:
            break
        if pid in used_ids or puzzle["fen"] in used_fens:
            continue
        used_ids.add(pid)
        used_fens.add(puzzle["fen"])
        selected.append((puzzle, info))
    return selected


def _explanation(board: chess.Board, info: dict, note: str = "") -> str:
    line = san_line(board, info["moves"])
    piece = PIECE_RU[info["piece"].piece_type]
    text = f"Решение: {line}."
    if info["mate"]:
        text += f" Ход {info['san_first']} сразу ставит мат: {piece} атакует короля, а поля отступления перекрыты."
    elif info["check"]:
        text += f" Ход {info['san_first']} объявляет шах."
    if info["captured"] is not None:
        text += (
            f" {piece.capitalize()} берёт {PIECE_ACC[info['captured'].piece_type]}"
            f" на {chess.square_name(info['first'].to_square)}."
        )
    if info["delta"] > 0:
        text += (
            f" В итоге перевес белых по материалу растёт на "
            f"{info['delta']} {_pawns_word(info['delta'])}."
        )
    if note:
        text += " " + note
    return text


def _puzzle_tasks(pool, used_fens, used_ids) -> dict[str, list[dict]]:
    plans = {
        "check": {
            "count": 9,
            "themes": ("mateIn2", "mateIn3", "oneMove", "check"),
            "rating": (500, 1400),
            "target": 900,
            # Мат - тоже шах, но урок про шах: берём именно не матующие шахи,
            # иначе инструкция «объявите шах» расходится с разбором «это мат».
            "require": lambda info, themes: info["check"] and not info["mate"],
            "motif": "Шах",
            "instruction": "Белые начинают. Сделайте ход, который объявляет шах чёрному королю.",
            "note": "Шах заставляет соперника реагировать и часто выигрывает темп.",
        },
        "escape-check": {
            "count": 8,
            "themes": ("defensiveMove", "endgame", "middlegame"),
            "rating": (500, 1700),
            "target": 1000,
            "require": lambda info, themes: info["board"].is_check()
            and info["piece"].piece_type == chess.KING,
            "motif": "Защита от шаха",
            "instruction": "Белые под шахом. Найдите безопасный ход королём.",
            "note": "Из-под шаха можно уйти королём, закрыть линию или взять атакующую фигуру.",
        },
        "piece-movement": {
            "count": 6,
            "themes": (),
            "rating": (600, 1600),
            "target": 1100,
        },
        "capture-and-defense": {
            "count": 8,
            "themes": (),
            "rating": (600, 1600),
            "target": 1100,
        },
        "special-rules": {
            "count": 6,
            "themes": ("castling", "enPassant"),
            "rating": (500, 1700),
            "target": 1000,
        },
        "back-rank-mate": {
            "count": 9,
            "themes": ("backRankMate", "mateIn1", "mateIn2"),
            "rating": (500, 1600),
            "target": 1000,
            # Тема mateIn1/mateIn2 сама по себе не гарантирует мат именно по
            # последней горизонтали: среди таких задач попадаются маты на
            # 7-й линии. Поэтому проверяем поле мата и фигуру.
            "require": lambda info, themes: _back_rank_mate(info),
            "motif": "Мат",
            "instruction": "Белые начинают. Поставьте мат по последней горизонтали.",
            "instruction_first": "Белые начинают. Найдите первый ход матовой атаки по последней горизонтали.",
            "note": "Король заперт на последней горизонтали своими фигурами - поэтому мат ладьёй или ферзём неотразим.",
        },
        "smothered-mate": {
            "count": 8,
            "themes": ("smotheredMate",),
            "rating": (500, 1700),
            "target": 1000,
            "require": lambda info, themes: _smothered_mate(info),
            "motif": "Спёртый мат",
            "instruction": "Белые начинают. Поставьте мат конём: король заперт своими фигурами.",
            "instruction_first": "Белые начинают. Найдите первый ход матовой атаки на запертого короля.",
            "note": "Король окружён своими фигурами, поэтому конь объявляет мат, а взять его нечем.",
        },
        "hanging-piece": {
            "count": 9,
            "themes": ("hangingPiece",),
            "rating": (500, 1600),
            "target": 1000,
            "require": lambda info, themes: info["captured"] is not None and info["delta"] > 0,
            "motif": "Незащищённая фигура",
            "instruction": "Белые начинают. Заберите фигуру соперника, которая осталась без защиты.",
        },
        "double-attack": {
            "count": 8,
            "themes": ("fork", "discoveredAttack"),
            "rating": (600, 1700),
            "target": 1100,
            "require": lambda info, themes: len(
                [item for item in info["attacked"] if VALUES[item[0]] >= 3]
            )
            >= 2,
            "motif": "Двойной удар",
            "instruction": "Белые начинают. Найдите ход, который создаёт сразу две угрозы.",
        },
        "fork": {
            "count": 9,
            "themes": ("fork",),
            "rating": (500, 1600),
            "target": 1000,
            "require": lambda info, themes: len(
                [item for item in info["attacked"] if VALUES[item[0]] >= 3]
            )
            >= 2,
            "motif": "Вилка",
            "instruction": "Белые начинают. Поставьте вилку: одна фигура должна напасть сразу на две цели.",
        },
        "pin": {
            "count": 9,
            "themes": ("pin",),
            "rating": (500, 1600),
            "target": 1000,
            "motif": "Связка",
            "instruction": "Белые начинают. Найдите ход, который связывает фигуру соперника.",
        },
        "capturing-defender": {
            "count": 8,
            "themes": ("capturingDefender",),
            "rating": (500, 1700),
            "target": 1100,
            "motif": "Уничтожение защитника",
            "instruction": "Белые начинают. Уберите фигуру, которая защищает важный пункт.",
        },
        "discovered-attack": {
            "count": 8,
            "themes": ("discoveredAttack",),
            "rating": (500, 1700),
            "target": 1100,
            "motif": "Открытое нападение",
            "instruction": "Белые начинают. Уйдите фигурой с линии, чтобы открыть удар другой фигуры.",
        },
        "discovered-check": {
            "count": 7,
            "themes": ("discoveredCheck",),
            "rating": (500, 1800),
            "target": 1100,
            "require": lambda info, themes: info["discovered_check"],
            "motif": "Вскрытый шах",
            "instruction": "Белые начинают. Освободите линию, чтобы объявить вскрытый шах.",
        },
        "combination": {
            "count": 9,
            "themes": ("sacrifice", "attraction", "clearance", "intermezzo", "deflection"),
            "rating": (700, 1800),
            "target": 1200,
            "require": lambda info, themes: len(info["moves"]) >= 4 and info["delta"] > 0,
            "motif": "Комбинация",
            "instruction": "Белые начинают. Найдите комбинацию с жертвой или отвлекающим ударом.",
        },
        "endgame-basics": {
            "count": 9,
            "themes": ("endgame",),
            "rating": (600, 1700),
            "target": 1100,
            # В базе Lichess тема endgame стоит и у тактических позиций с
            # ферзями. Для урока об эндшпиле берём только малый материал.
            "require": lambda info, themes: material(info["board"], chess.WHITE)
            + material(info["board"], chess.BLACK)
            <= 18,
            "motif": "Эндшпиль",
            "instruction": "Белые начинают. Найдите лучший ход в эндшпиле.",
        },
        "pawn-square": {
            "count": 8,
            "themes": ("pawnEndgame", "advancedPawn"),
            "rating": (600, 1700),
            "target": 1000,
            # Правило квадрата - про чистый пешечный эндшпиль: никаких
            # ладей и ферзей, иначе задача не про бег пешки.
            "require": lambda info, themes: info["piece"].piece_type == chess.PAWN
            and _only_kings_and_pawns(info["board"]),
            "motif": "Пешечный бег",
            "instruction": "Белые начинают. Продвиньте пешку: в пешечном окончании важен каждый темп.",
        },
        "king-and-pawn": {
            "count": 8,
            "themes": ("pawnEndgame",),
            "rating": (500, 1700),
            "target": 1000,
            "require": lambda info, themes: info["piece"].piece_type == chess.KING,
            "motif": "Король и пешка",
            "instruction": "Белые начинают. Найдите точный ход королём в пешечном окончании.",
        },
        "doubled-pawns": {
            "count": 9,
            "themes": ("endgame", "middlegame"),
            "rating": (600, 1800),
            "target": 1100,
            # Ход должен использовать именно слабость сдвоенных пешек:
            # нападать на них или забирать одну из них.
            "require": lambda info, themes: _uses_doubled_pawns(info),
            "motif": "Сдвоенные пешки",
            "instruction": "Белые начинают. Заберите одну из сдвоенных пешек соперника.",
            "note": "Сдвоенные пешки не прикрывают друг друга - одну из них можно забрать.",
        },
        "king-activity": {
            "count": 8,
            "themes": ("endgame",),
            "rating": (600, 1800),
            "target": 1100,
            "require": lambda info, themes: info["piece"].piece_type == chess.KING,
            "motif": "Активность короля",
            "instruction": "Белые начинают. Активизируйте короля - в эндшпиле он сильная фигура.",
        },
        "material-advantage": {
            "count": 8,
            "themes": ("advantage", "crushing", "hangingPiece"),
            "rating": (600, 1800),
            "target": 1100,
            "require": lambda info, themes: info["captured"] is not None and info["delta"] > 0,
            "motif": "Материал",
            "instruction": "Белые начинают. Заберите материал и закрепите перевес.",
        },
        "positional-advantage": {
            "count": 8,
            "themes": ("quietMove", "advantage"),
            "rating": (700, 1900),
            "target": 1200,
            "require": lambda info, themes: info["quiet"],
            "motif": "Улучшение позиции",
            "instruction": "Белые начинают. Найдите спокойный ход, который улучшает позицию.",
        },
        "open-files": {
            "count": 8,
            "themes": ("middlegame", "rookEndgame"),
            "rating": (700, 1900),
            "target": 1200,
            "require": lambda info, themes: info["piece"].piece_type
            in (chess.ROOK, chess.QUEEN)
            and _moves_to_open_file(info),
            "motif": "Открытая линия",
            "instruction": "Белые начинают. Заняните открытую линию тяжёлой фигурой.",
        },
        "good-bad-pieces": {
            "count": 8,
            "themes": ("middlegame", "endgame"),
            "rating": (600, 1800),
            "target": 1100,
            "require": lambda info, themes: info["piece"].piece_type
            in (chess.KNIGHT, chess.BISHOP)
            and _is_central(info["first"].to_square),
            "motif": "Хорошая фигура",
            "instruction": "Белые начинают. Переведите фигуру ближе к центру.",
        },
        "middlegame-mistakes": {
            "count": 9,
            "themes": ("defensiveMove", "exposedKing", "kingsideAttack"),
            "rating": (700, 1900),
            "target": 1200,
            "motif": "Защита",
            "instruction": "Белые начинают. Найдите ход, который устраняет угрозу соперника.",
        },
    }

    tasks: dict[str, list[dict]] = {}

    for slug, spec in plans.items():
        if slug in ("piece-movement", "capture-and-defense", "special-rules"):
            continue
        chosen = pick(pool, spec, used_fens, used_ids)
        lesson_tasks = []
        for puzzle, info in chosen:
            board = info["board"]
            piece = info["piece"]
            dest = chess.square_name(info["first"].to_square)
            motif = spec["motif"]
            instruction = spec.get(
                "instruction",
                "Белые начинают. Найдите лучший ход в позиции из реальной партии.",
            )
            # Заголовок про мат уместен только в матовых темах. Иначе
            # «Мат в 2 хода» появлялся в уроках про активность короля и
            # защиту от шаха, где белые как раз спасаются.
            if motif in ("Мат", "Спёртый мат"):
                if info["mate"]:
                    title = "Мат в один ход"
                else:
                    # Решение длиннее одного хода: задача просит первый ход
                    # матовой атаки, поэтому обещать мат прямо сейчас нельзя.
                    moves_to_mate = max(2, (len(info["moves"]) + 1) // 2)
                    title = f"Мат в {moves_to_mate} {_moves_word(moves_to_mate)}"
                    instruction = spec.get("instruction_first", instruction)
            else:
                title = f"{motif}: {PIECE_INSTR[piece.piece_type]} на {dest}"
            difficulty = 1 if puzzle.get("rating", 0) < 1000 else 2 if puzzle.get("rating", 0) < 1500 else 3
            lesson_tasks.append(
                {
                    "task_type": "make_move",
                    "title": title,
                    "instruction": instruction,
                    "fen": puzzle["fen"],
                    "source_square": chess.square_name(info["first"].from_square),
                    "payload": {
                        "piece": piece.symbol().upper(),
                        "mode": "accepted_moves",
                        "accepted_moves": [info["first"].uci()],
                    },
                    "explanation": _explanation(board, info, spec.get("note", "")),
                    "difficulty": difficulty,
                }
            )
        if lesson_tasks:
            tasks[slug] = lesson_tasks
    return tasks


def _is_central(square: int) -> bool:
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return 2 <= file <= 5 and 2 <= rank <= 5


def _only_kings_and_pawns(board: chess.Board) -> bool:
    return all(
        piece.piece_type in (chess.KING, chess.PAWN)
        for piece in board.piece_map().values()
    )


def _mating_piece(info: dict) -> chess.Piece | None:
    """Фигура, которая ставит мат последним ходом решения."""
    if not info["final_mate"]:
        return None
    last = info["moves"][-1]
    return info["final"].piece_at(last.to_square)


def _back_rank_mate(info: dict) -> bool:
    """Мат по последней горизонтали: ладья или ферзь матуют на 8-й линии."""
    piece = _mating_piece(info)
    if piece is None or piece.piece_type not in (chess.ROOK, chess.QUEEN):
        return False
    return chess.square_rank(info["moves"][-1].to_square) == 7


def _smothered_mate(info: dict) -> bool:
    """Спёртый мат: последний ход делает конь."""
    piece = _mating_piece(info)
    return piece is not None and piece.piece_type == chess.KNIGHT


def _has_doubled_pawns(board: chess.Board) -> bool:
    for color in (chess.WHITE, chess.BLACK):
        files = [
            chess.square_file(square)
            for square, piece in board.piece_map().items()
            if piece.color == color and piece.piece_type == chess.PAWN
        ]
        if len(files) != len(set(files)):
            return True
    return False


def _doubled_pawn_files(board: chess.Board, color: bool) -> set[int]:
    files = [
        chess.square_file(square)
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type == chess.PAWN
    ]
    return {file for file in files if files.count(file) >= 2}


def _uses_doubled_pawns(info: dict) -> bool:
    """Ход забирает одну из сдвоенных пешек соперника."""
    board = info["board"]
    files = _doubled_pawn_files(board, chess.BLACK)
    if not files:
        return False
    captured = info["captured"]
    return (
        captured is not None
        and captured.color == chess.BLACK
        and captured.piece_type == chess.PAWN
        and chess.square_file(info["first"].to_square) in files
    )


def _moves_to_open_file(info: dict) -> bool:
    board = info["board"]
    square = info["first"].to_square
    file = chess.square_file(square)
    for color in (chess.WHITE, chess.BLACK):
        for sq, piece in board.piece_map().items():
            if (
                piece.color == color
                and piece.piece_type == chess.PAWN
                and chess.square_file(sq) == file
            ):
                return False
    return True


# --- Базовые темы: вопросы и задания по реальным позициям -----------------


def _choose(title, instruction, fen, options, correct, explanation, difficulty=1):
    return {
        "task_type": "choose_option",
        "title": title,
        "instruction": instruction,
        "fen": fen,
        "source_square": None,
        "payload": {
            "options": [
                {"id": option_id, "label": label} for option_id, label in options
            ],
            "correct_option": correct,
        },
        "explanation": explanation,
        "difficulty": difficulty,
    }


def _move(title, instruction, fen, source, explanation, accepted, difficulty=1):
    board = chess.Board(fen)
    piece = board.piece_at(chess.parse_square(source))
    return {
        "task_type": "make_move",
        "title": title,
        "instruction": instruction,
        "fen": fen,
        "source_square": source,
        "payload": {
            "piece": piece.symbol().upper(),
            "mode": "accepted_moves",
            "accepted_moves": accepted,
        },
        "explanation": explanation,
        "difficulty": difficulty,
    }


def _select(title, instruction, fen, source, explanation, mode="all_legal_moves", difficulty=1):
    board = chess.Board(fen)
    piece = board.piece_at(chess.parse_square(source))
    return {
        "task_type": "select_squares",
        "title": title,
        "instruction": instruction,
        "fen": fen,
        "source_square": source,
        "payload": {"piece": piece.symbol().upper(), "mode": mode},
        "explanation": explanation,
        "difficulty": difficulty,
    }


def _square_color(square: str) -> str:
    file = chess.parse_square(square) % 8
    rank = chess.parse_square(square) // 8
    return "светлое" if (file + rank) % 2 == 1 else "тёмное"


def _nav(
    title: str,
    fen: str,
    source: str,
    target: str,
    instruction: str,
    explanation: str,
    difficulty: int = 1,
) -> dict:
    """Задание «переведите фигуру с поля X на поле Y» - тренировка координат."""
    move = chess.Move.from_uci(source + target)
    return _move(
        title,
        instruction,
        fen,
        source,
        explanation,
        [move.uci()],
        difficulty=difficulty,
    )


def _navigation_tasks(fen: str, rows) -> list[dict]:
    """Строит задания навигации, отбрасывая ходы, нелегальные в позиции."""
    board = chess.Board(fen)
    tasks = []
    for source, target, label, explanation in rows:
        move = chess.Move.from_uci(source + target)
        if move not in board.legal_moves:
            continue
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        tasks.append(
            _nav(
                f"Навигация: {label}",
                fen,
                source,
                target,
                f"Переведите {PIECE_ACC[piece.piece_type]} с {source} на {target}.",
                explanation,
            )
        )
    return tasks


def _pawns_word(count: int) -> str:
    count = abs(count)
    if count % 10 == 1 and count % 100 != 11:
        return "пешку"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "пешки"
    return "пешек"


def _moves_word(count: int) -> str:
    count = abs(count)
    if count % 10 == 1 and count % 100 != 11:
        return "ход"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "хода"
    return "ходов"


MATERIAL_TITLES = (
    "Материальный перевес",
    "Считаем материал",
    "У кого больше?",
    "Баланс материала",
    "Сравните материал",
    "Лишний материал",
    "Счёт по пешкам",
    "Материал на доске",
)


def _material_tasks(pool, used_fens) -> list[dict]:
    """Вопросы «у кого перевес» по реальным позициям.

    Берём финальную позицию решения задачи: в задачах Lichess игрок часто
    отыгрывает материал из худшего положения, поэтому считать перевес в
    начальной позиции бессмысленно (там решающий почти всегда позади). Линия
    всегда заканчивается ходом белых, поэтому ход передаём белым. Позиции,
    где чёрный король под шахом, отбрасываем: после смены стороны хода они
    были бы нелегальными. Счёт и пояснение считаются по доске, так что текст
    всегда совпадает с картинкой.
    """
    wanted = ["equal", "white", "black", "white", "black", "equal", "white", "black"]
    buckets: dict[str, list[tuple[str, int]]] = {"equal": [], "white": [], "black": []}
    ordered = sorted(
        pool, key=lambda item: (int(item.get("rating") or 0), item.get("id") or "")
    )
    for puzzle in ordered:
        if puzzle["fen"] in used_fens:
            continue
        info = facts(puzzle)
        if info is None or info["solver_mated"]:
            continue
        board = info["final"]
        if board.is_checkmate() or board.is_stalemate() or board.is_check():
            continue
        white_total = material(board, chess.WHITE)
        black_total = material(board, chess.BLACK)
        # Нищие эндшпили (две пешки против двух) для урока о ценности фигур
        # бесполезны: считать там нечего. Нужна позиция с фигурами.
        if white_total + black_total < 20:
            continue
        diff = white_total - black_total
        if diff == 0:
            key = "equal"
        elif diff > 0:
            key = "white"
        else:
            key = "black"
        flipped = board.copy()
        flipped.turn = chess.WHITE
        buckets[key].append((flipped.fen(), diff))

    tasks: list[dict] = []
    for index, side in enumerate(wanted):
        bucket = buckets[side]
        while bucket:
            final_fen, diff = bucket.pop(0)
            if final_fen in used_fens:
                continue
            used_fens.add(final_fen)
            board = chess.Board(final_fen)
            white_total = material(board, chess.WHITE)
            black_total = material(board, chess.BLACK)
            if diff == 0:
                correct = "equal"
                explanation = (
                    f"Материал равен: у белых {white_total}, у чёрных {black_total}."
                )
            elif diff > 0:
                correct = "white"
                explanation = (
                    f"У белых перевес на {diff} {_pawns_word(diff)}: "
                    f"{white_total} против {black_total}."
                )
            else:
                correct = "black"
                explanation = (
                    f"У чёрных перевес на {-diff} {_pawns_word(diff)}: "
                    f"{black_total} против {white_total}."
                )
            tasks.append(
                _choose(
                    MATERIAL_TITLES[index % len(MATERIAL_TITLES)],
                    "У кого материальный перевес в этой позиции?",
                    final_fen,
                    [
                        ("equal", "Материал примерно равен"),
                        ("white", "У белых"),
                        ("black", "У чёрных"),
                    ],
                    correct,
                    explanation,
                )
            )
            break
    return tasks


def basic_tasks(pool, used_fens) -> dict[str, list[dict]]:
    start = chess.STARTING_FEN
    mid = "r2q1rk1/pp3ppp/2n1bn2/2bp4/8/1PN1PN2/PB1PBPPP/R2Q1RK1 w - - 0 10"
    result: dict[str, list[dict]] = {}

    result["board-coordinates"] = [
        _choose(
            "Цвет поля d5",
            "Какого цвета поле d5?",
            mid,
            [("light", "Светлого"), ("dark", "Тёмного")],
            "light" if _square_color("d5") == "светлое" else "dark",
            f"Поле d5 {_square_color('d5')}.",
        ),
        _choose(
            "Соседнее поле",
            "Какое поле находится справа от e4?",
            mid,
            [("f4", "f4"), ("d4", "d4"), ("e5", "e5"), ("e3", "e3")],
            "f4",
            "Справа от e4 находится поле f4: вертикали идут слева направо от a до h.",
        ),
        _choose(
            "Вертикаль поля",
            "На какой вертикали находится поле f6?",
            mid,
            [("f", "f"), ("6", "6"), ("g", "g"), ("e", "e")],
            "f",
            "Первая буква записи поля - вертикаль, здесь это f.",
        ),
        _choose(
            "Горизонталь поля",
            "На какой горизонтали находится поле b7?",
            mid,
            [("7", "7"), ("b", "b"), ("6", "6"), ("8", "8")],
            "7",
            "Цифра в записи поля - горизонталь, здесь это 7.",
        ),
        _choose(
            "Фигура на c5",
            "Какая фигура стоит на поле c5?",
            mid,
            [
                ("bishop", "Чёрный слон"),
                ("knight", "Чёрный конь"),
                ("rook", "Чёрная ладья"),
                ("queen", "Чёрный ферзь"),
            ],
            "bishop",
            "На c5 стоит чёрный слон.",
        ),
    ] + _navigation_tasks(
        mid,
        [
            ("c3", "e4", "конь c3-e4", "Конь с c3 прыгает через поле d5 на e4."),
            ("b2", "a3", "слон b2-a3", "Слон b2 идёт по диагонали b2-a3."),
            ("d1", "c2", "ферзь d1-c2", "Ферзь d1 идёт по диагонали на c2."),
            ("a1", "c1", "ладья a1-c1", "Ладья идёт по первой горизонтали через пустое b1."),
            ("f1", "e1", "ладья f1-e1", "Ладья f1 сдвигается на соседнее поле e1."),
            ("g1", "h1", "король g1-h1", "Король делает один шаг вправо, на h1."),
            ("a2", "a4", "пешка a2-a4", "Пешка с начального поля может прыгнуть через клетку: a2-a4."),
            ("d2", "d3", "пешка d2-d3", "Пешка d2 стоит перед ферзём и идёт на d3."),
            ("h2", "h3", "пешка h2-h3", "Пешка h2 идёт на одно поле вперёд, на h3."),
        ],
    )

    result["piece-setup"] = [
        _choose(
            "Белый ферзь",
            "На каком поле стоит белый ферзь в начальной позиции?",
            start,
            [("d1", "d1"), ("e1", "e1"), ("c1", "c1"), ("d8", "d8")],
            "d1",
            "Белый ферзь стоит на d1, король - на e1.",
        ),
        _choose(
            "Соседи слона c1",
            "Какие фигуры стоят рядом с белым слоном c1?",
            start,
            [
                ("knight_queen", "Конь b1 и ферзь d1"),
                ("rooks", "Две ладьи"),
                ("pawns", "Только пешки"),
                ("king", "Король e1 и ферзь d1"),
            ],
            "knight_queen",
            "На b1 стоит конь, на d1 - ферзь, между ними слон c1.",
        ),
        _choose(
            "Всего фигур",
            "Сколько фигур и пешек у каждой стороны в начале партии?",
            start,
            [("16", "16"), ("8", "8"), ("12", "12"), ("20", "20")],
            "16",
            "Восемь фигур и восемь пешек - всего 16 единиц у каждой стороны.",
        ),
    ] + _navigation_tasks(
        start,
        [
            ("g1", "f3", "конь g1-f3", "Конь с g1 встаёт на f3: это поле в центре рядом с королём."),
            ("b1", "c3", "конь b1-c3", "Конь с b1 идёт на c3 - ближе к центру."),
            ("g1", "h3", "конь g1-h3", "Конь может пойти и на край: g1-h3."),
            ("b1", "a3", "конь b1-a3", "С b1 конь достаёт и поле a3 на краю доски."),
            ("e2", "e3", "пешка e2-e3", "Пешка e2 может шагнуть вперёд на одно поле - на e3."),
            ("d2", "d4", "пешка d2-d4", "Пешка d2 стоит перед ферзём и идёт на d4."),
            ("c2", "c3", "пешка c2-c3", "Пешка c2 идёт на одно поле вперёд, на c3."),
            ("f2", "f3", "пешка f2-f3", "Пешка f2 идёт на f3, освобождая дорогу слону f1."),
            ("h2", "h4", "пешка h2-h4", "Крайняя пешка h2 может шагнуть сразу на h4."),
        ],
    )

    # Базовые вопросы о ценности фигур уже есть в курсе, поэтому здесь
    # добавляем только позиции из партий: посчитать материал на доске.
    result["piece-values"] = _material_tasks(pool, used_fens)

    # Нотация: читаем запись и делаем ход на доске, SAN считается python-chess.
    notation_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    notation_tasks = []
    notation_questions = [
        ("Nc3", "b1", "c3"),
        ("Bc4", "f1", "c4"),
        ("Bb5", "f1", "b5"),
        ("Be2", "f1", "e2"),
        ("d4", "d2", "d4"),
        ("d3", "d2", "d3"),
        ("a3", "a2", "a3"),
        ("h3", "h2", "h3"),
        ("Qe2", "d1", "e2"),
    ]
    for san, source, target in notation_questions:
        move = chess.Move.from_uci(source + target)
        if move not in notation_board.legal_moves:
            continue
        if notation_board.san(move) != san:
            continue
        notation_tasks.append(
            _move(
                f"Запись хода: {san}",
                f"Сделайте ход, который в записи выглядит как {san}.",
                notation_board.fen(),
                source,
                f"{san} - это ход с {source} на {target}.",
                [move.uci()],
            )
        )
    notation_tasks.extend(
        [
            _choose(
                "Что означает x",
                "Что означает символ x в шахматной записи?",
                start,
                [
                    ("capture", "Взятие"),
                    ("check", "Шах"),
                    ("mate", "Мат"),
                    ("castle", "Рокировка"),
                ],
                "capture",
                "x обозначает взятие: например, exd5.",
            ),
            _choose(
                "Что означает +",
                "Что означает знак + в шахматной записи?",
                start,
                [
                    ("check", "Шах"),
                    ("capture", "Взятие"),
                    ("mate", "Мат"),
                    ("promotion", "Превращение пешки"),
                ],
                "check",
                "+ после хода означает шах.",
            ),
            _choose(
                "Что означает O-O",
                "Что означает запись O-O?",
                start,
                [
                    ("short", "Короткая рокировка"),
                    ("long", "Длинная рокировка"),
                    ("queen", "Ход ферзём"),
                    ("castle_long", "Рокировка в длинную сторону"),
                ],
                "short",
                "O-O - короткая рокировка, O-O-O - длинная.",
            ),
            _choose(
                "Что означает #",
                "Что означает знак # в шахматной записи?",
                start,
                [
                    ("mate", "Мат"),
                    ("check", "Шах"),
                    ("capture", "Взятие"),
                    ("draw", "Ничья"),
                ],
                "mate",
                "# ставится после матующего хода.",
            ),
        ]
    )
    result["notation"] = notation_tasks
    return result


def select_tasks(pool, used_fens) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {"piece-movement": [], "capture-and-defense": []}
    wanted = {
        "piece-movement": {"count": 6, "kinds": (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)},
        "capture-and-defense": {"count": 8, "kinds": (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)},
    }
    made = {"piece-movement": 0, "capture-and-defense": 0}
    for puzzle in sorted(pool, key=lambda item: (int(item.get("rating") or 0), item.get("id") or "")):
        if puzzle["fen"] in used_fens:
            continue
        rating = int(puzzle.get("rating") or 0)
        if rating < 600 or rating > 1700:
            continue
        board = chess.Board(puzzle["fen"])
        if board.turn != chess.WHITE:
            continue
        for square, piece in board.piece_map().items():
            if piece.color != chess.WHITE:
                continue
            if piece.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                continue
            moves = [m for m in board.legal_moves if m.from_square == square]
            captures = [m for m in moves if board.is_capture(m)]
            if made["piece-movement"] < wanted["piece-movement"]["count"] and len(moves) >= 4:
                dests = sorted(chess.square_name(m.to_square) for m in moves)
                name = chess.square_name(square)
                result["piece-movement"].append(
                    _select(
                        f"Ходы {PIECE_GEN[piece.piece_type]} {name}",
                        f"Отметьте все клетки, на которые может перейти {_piece_phrase(name, piece.piece_type)}.",
                        puzzle["fen"],
                        name,
                        f"{PIECE_RU[piece.piece_type].capitalize()} {name} может пойти на "
                        + ", ".join(dests)
                        + ".",
                        difficulty=1,
                    )
                )
                made["piece-movement"] += 1
                used_fens.add(puzzle["fen"])
                break
            if made["capture-and-defense"] < wanted["capture-and-defense"]["count"] and captures:
                dests = sorted(
                    f"{chess.square_name(m.to_square)} ({PIECE_RU[board.piece_at(m.to_square).piece_type]})"
                    for m in captures
                    if board.piece_at(m.to_square) is not None
                )
                name = chess.square_name(square)
                result["capture-and-defense"].append(
                    _select(
                        f"Взятия {PIECE_GEN[piece.piece_type]} {name}",
                        f"Отметьте клетки, на которых {_piece_phrase(name, piece.piece_type)} может взять фигуру противника.",
                        puzzle["fen"],
                        name,
                        f"{PIECE_RU[piece.piece_type].capitalize()} {name} может взять на "
                        + ", ".join(dests)
                        + ".",
                        mode="capture_squares",
                        difficulty=1,
                    )
                )
                made["capture-and-defense"] += 1
                used_fens.add(puzzle["fen"])
                break
        if all(made[k] >= wanted[k]["count"] for k in wanted):
            break
    return result


def special_tasks(pool, used_fens) -> dict[str, list[dict]]:
    """Рокировка и взятие на проходе из реальных партий.

    Берём позиции, где первым ходом решения в партии была именно рокировка
    или взятие на проходе, а не любой формально легальный такой ход.
    """
    castling: list[dict] = []
    en_passant: list[dict] = []
    for puzzle in sorted(pool, key=lambda item: (int(item.get("rating") or 0), item.get("id") or "")):
        if puzzle["fen"] in used_fens:
            continue
        if len(castling) >= 6 and len(en_passant) >= 4:
            break
        try:
            board = chess.Board(puzzle["fen"])
        except ValueError:
            continue
        if board.turn != chess.WHITE:
            continue
        info = facts(puzzle)
        if info is None or info["solver_mated"]:
            continue
        move = info["first"]
        if len(castling) < 6 and board.is_castling(move):
            side = "короткая" if board.is_kingside_castling(move) else "длинная"
            castling.append(
                _move(
                    f"Рокировка из партии: {side}",
                    "Сделайте рокировку: король уходит из центра, ладья вступает в игру.",
                    puzzle["fen"],
                    chess.square_name(move.from_square),
                    _explanation(board, info, "Рокировка завершает развитие и делает короля безопаснее."),
                    [move.uci()],
                    difficulty=1,
                )
            )
            used_fens.add(puzzle["fen"])
            continue
        if len(en_passant) < 4 and board.is_en_passant(move):
            en_passant.append(
                _move(
                    "Взятие на проходе",
                    "Возьмите пешку соперника на проходе.",
                    puzzle["fen"],
                    chess.square_name(move.from_square),
                    _explanation(
                        board,
                        info,
                        "Пешка берёт через поле, которое только что перешагнула пешка соперника.",
                    ),
                    [move.uci()],
                    difficulty=2,
                )
            )
            used_fens.add(puzzle["fen"])
    if not en_passant:
        # Редкий случай: соберём учебную позицию, проверенную python-chess.
        fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        board = chess.Board(fen)
        move = chess.Move.from_uci("e5f6")
        if move in board.legal_moves and board.is_en_passant(move):
            en_passant.append(
                _move(
                    "Взятие на проходе",
                    "Возьмите пешку f5 на проходе.",
                    fen,
                    "e5",
                    "exf6 - взятие на проходе: пешка e5 берёт пешку f5, которая перешагнула поле f6.",
                    ["e5f6"],
                    difficulty=2,
                )
            )
    return {"special-rules": castling + en_passant}


def stalemate_tasks() -> list[dict]:
    """Учебные позиции, где ход белых создаёт пат (проверено python-chess)."""
    raw = [
        ("7k/8/6K1/8/8/8/5Q2/8 w - - 0 1", "f2f7", "Qf7 создаёт пат: король h8 не под шахом, но поля g7, g8 и h7 под контролем."),
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6c7", "Qc7 создаёт пат: у короля a8 нет ни одного хода, а шаха нет."),
        ("k7/8/8/2Q5/8/8/8/1K6 w - - 0 1", "c5b6", "Qb6 создаёт пат: поля a7, b7 и b8 под контролем, шаха нет."),
        ("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", "f1f7", "Qf7 создаёт пат: король h8 заперт, но не атакован."),
        ("3k4/8/3K4/2Q5/8/8/8/8 w - - 0 1", "c5c6", "Qc6 создаёт пат: у короля d8 нет безопасных полей, шаха нет."),
        ("8/8/8/8/8/3Q4/2K5/k7 w - - 0 1", "d3b3", "Qb3 создаёт пат: король a1 не под шахом, но не может двинуться."),
        ("k7/8/K7/8/8/8/8/2Q5 w - - 0 1", "c1c7", "Qc7 создаёт пат: поля a7, b7 и b8 под контролем, шаха нет."),
        ("k7/8/K7/8/8/8/8/5Q2 w - - 0 1", "f1b5", "Qb5 создаёт пат: король a8 заперт на краю, но не атакован."),
        ("k7/8/K7/8/8/8/8/3Q4 w - - 0 1", "d1d6", "Qd6 создаёт пат: поля a7, b7 и b8 контролируются ферзём и королём."),
    ]
    tasks = []
    for fen, uci, explanation in raw:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if not board.is_stalemate():
            continue
        tasks.append(
            _move(
                "Создайте пат",
                "Сделайте ход, после которого у чёрных не останется ни одного легального хода, но не будет шаха.",
                fen,
                chess.square_name(move.from_square),
                explanation,
                [uci],
                difficulty=2,
            )
        )
    return tasks


def king_mate_tasks() -> dict[str, list[dict]]:
    """Мат ферзём и ладьёй: классические позиции с проверенным матом."""
    queen = [
        ("7k/Q7/6K1/8/8/8/8/8 w - - 0 1", "a7g7", "Ферзь встаёт на g7 рядом с королём: поле h8 атаковано, а отступить некуда."),
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6b7", "Ферзь на b7 отрезает короля a8 по вертикали b и диагонали; король c6 держит a7."),
        ("3k4/8/3K4/8/8/8/8/1Q6 w - - 0 1", "b1b8", "Ферзь на b8 объявляет мат по восьмой горизонтали: поля c7, d7 и e7 держит король d6."),
        ("8/8/8/8/8/5K2/1Q6/7k w - - 0 1", "b2g2", "Ферзь на g2 отрезает короля h1: поля g1 и h2 под контролем ферзя и короля f3."),
        ("7k/8/6K1/8/8/8/8/1Q6 w - - 0 1", "b1b8", "Ферзь на b8 ставит мат по последней горизонтали; король g6 держит g7 и h7."),
        ("k7/8/1K6/8/8/8/8/6Q1 w - - 0 1", "g1g8", "Ферзь на g8 ставит мат по восьмой горизонтали; король b6 держит a7 и b7."),
        ("k7/8/K7/8/8/8/8/6Q1 w - - 0 1", "g1g8", "Ферзь на g8 ставит мат по восьмой горизонтали; король a6 держит a7 и b7."),
        ("k7/8/K7/8/8/8/8/3Q4 w - - 0 1", "d1d8", "Ферзь на d8 ставит мат по восьмой горизонтали; король a6 держит a7 и b7."),
    ]
    rook = [
        ("7k/8/6K1/8/8/8/8/5R2 w - - 0 1", "f1f8", "Ладья на f8 ставит мат по последней горизонтали: король g6 держит g7 и h7."),
        ("k7/8/1K6/8/8/8/8/7R w - - 0 1", "h1h8", "Ладья на h8 ставит мат по восьмой горизонтали; король b6 держит a7 и b7."),
        ("3k4/8/3K4/8/8/8/8/R7 w - - 0 1", "a1a8", "Ладья на a8 ставит мат по восьмой горизонтали; король d6 держит c7, d7 и e7."),
        ("6k1/8/6K1/8/8/8/8/R7 w - - 0 1", "a1a8", "Ладья на a8 ставит мат по последней горизонтали; король g6 держит g7, h7 и f7."),
        ("7k/8/6K1/8/8/8/8/4R3 w - - 0 1", "e1e8", "Ладья на e8 ставит мат по последней горизонтали; король g6 держит g7 и h7."),
        ("k7/8/K7/8/8/8/8/3R4 w - - 0 1", "d1d8", "Ладья на d8 ставит мат по восьмой горизонтали; король a6 держит a7 и b7."),
        ("k7/8/K7/8/8/8/6R1/8 w - - 0 1", "g2g8", "Ладья на g8 ставит мат по восьмой горизонтали; король a6 держит a7 и b7."),
        ("k7/8/K7/8/8/8/8/7R w - - 0 1", "h1h8", "Ладья на h8 ставит мат по восьмой горизонтали; король a6 держит a7 и b7."),
        ("k7/8/K7/8/8/8/8/2R5 w - - 0 1", "c1c8", "Ладья на c8 ставит мат по восьмой горизонтали; король a6 держит a7 и b7."),
    ]

    def build(raw, title, instruction):
        tasks = []
        for fen, uci, explanation in raw:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                continue
            board.push(move)
            if not board.is_checkmate():
                continue
            tasks.append(
                _move(
                    title,
                    instruction,
                    fen,
                    chess.square_name(move.from_square),
                    explanation,
                    [uci],
                    difficulty=2,
                )
            )
        return tasks

    return {
        "king-queen-mate": build(
            queen,
            "Мат ферзём и королём",
            "Мат в один ход: поставьте ферзя так, чтобы у чёрного короля не осталось полей.",
        ),
        "king-rook-mate": build(
            rook,
            "Мат ладьёй и королём",
            "Мат в один ход: отрежьте короля ладьёй по последней горизонтали.",
        ),
    }


def opposition_tasks() -> list[dict]:
    """Пешечные окончания: король занимает ключевое поле (проверено Stockfish)."""
    raw = [
        ("8/4k3/8/4K3/4P3/8/8/8 w - - 0 1", "e5f5", "Kf5 - король обходит соперника и забирает оппозицию."),
        ("8/8/8/2k5/8/2K5/2P5/8 w - - 0 1", "c3b3", "Kb3 - король занимает ключевое поле и вытесняет чёрного короля."),
        ("8/8/8/2k5/8/2K5/1P6/8 w - - 0 1", "c3b3", "Kb3 - король становится рядом с пешкой и получает оппозицию."),
        ("8/8/2k5/8/2K5/2P5/8/8 w - - 0 1", "c4d4", "Kd4 - король захватывает центр и оттесняет соперника."),
        ("8/8/8/3k4/8/3K4/3P4/8 w - - 0 1", "d3e3", "Ke3 - король занимает ключевое поле рядом с пешкой."),
        ("8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", "e1d2", "Kd2 - король идёт вперёд и берёт оппозицию."),
        ("8/8/8/3k4/8/2K5/3P4/8 w - - 0 1", "c3d3", "Kd3 занимает оппозицию и вытесняет чёрного короля - пешка d2 проходит."),
        ("8/8/8/2k5/8/1K6/2P5/8 w - - 0 1", "b3c3", "Kc3 подходит к пешке и получает оппозицию: чёрный король не успевает её задержать."),
    ]
    tasks = []
    for fen, uci, explanation in raw:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        tasks.append(
            _move(
                "Оппозиция королей",
                "Белые начинают. Найдите ход королём, который захватывает оппозицию.",
                fen,
                chess.square_name(move.from_square),
                explanation,
                [uci],
                difficulty=2,
            )
        )
    return tasks


def _pawn_islands(board: chess.Board, color: bool) -> int:
    files = sorted(
        {
            chess.square_file(square)
            for square, piece in board.piece_map().items()
            if piece.color == color and piece.piece_type == chess.PAWN
        }
    )
    if not files:
        return 0
    islands = 1
    for left, right in zip(files, files[1:]):
        if right - left > 1:
            islands += 1
    return islands


def _pawn_squares(board: chess.Board, color: bool) -> list[int]:
    return [
        square
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type == chess.PAWN
    ]


def _passed_squares(board: chess.Board, color: bool) -> list[int]:
    """Пешки, перед которыми нет пешек соперника на своей и соседних вертикалях."""
    enemy = not color
    enemy_squares = _pawn_squares(board, enemy)
    result = []
    for square in _pawn_squares(board, color):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        blocked = False
        for other in enemy_squares:
            other_file = chess.square_file(other)
            other_rank = chess.square_rank(other)
            if abs(other_file - file) > 1:
                continue
            if color == chess.WHITE and other_rank > rank:
                blocked = True
                break
            if color == chess.BLACK and other_rank < rank:
                blocked = True
                break
        if not blocked:
            result.append(square)
    return result


def _isolated_squares(board: chess.Board, color: bool) -> list[int]:
    files = {chess.square_file(square) for square in _pawn_squares(board, color)}
    result = []
    for square in _pawn_squares(board, color):
        file = chess.square_file(square)
        if file - 1 not in files and file + 1 not in files:
            result.append(square)
    return result


def pawn_structure_tasks(pool, used_fens, used_ids) -> list[dict]:
    """Задания на пешечную структуру из реальных партий.

    Берём позиции, где ход белых - ход пешкой, а в структуре есть
    характерная особенность: сдвоенные, изолированные или проходные пешки.
    """
    notes = {
        "doubled": "Сдвоенные пешки не защищают друг друга и часто становятся слабостью.",
        "passed": "Проходную пешку важно продвигать: она сковывает фигуры соперника.",
        "isolated": "Изолированная пешка не защищена пешкой и требует постоянной опеки.",
        "islands": "Чем меньше пешечных островков, тем крепче пешечная структура.",
    }
    kinds = ["doubled", "doubled", "doubled", "passed", "passed", "passed",
             "isolated", "isolated", "islands"]
    tasks: list[dict] = []
    for puzzle in sorted(
        pool, key=lambda item: (int(item.get("rating") or 0), item.get("id") or "")
    ):
        if not kinds:
            break
        pid = puzzle.get("id")
        fen = puzzle["fen"]
        if pid in used_ids or fen in used_fens:
            continue
        rating = int(puzzle.get("rating") or 0)
        if rating < 600 or rating > 1800:
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.turn != chess.WHITE:
            continue
        info = facts(puzzle)
        if info is None:
            continue
        piece = board.piece_at(info["first"].from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            continue
        for kind in list(kinds):
            if kind == "doubled" and not _has_doubled_pawns(board):
                continue
            if kind == "passed" and not (
                _passed_squares(board, chess.WHITE)
                or _passed_squares(board, chess.BLACK)
            ):
                continue
            if kind == "isolated" and not (
                _isolated_squares(board, chess.WHITE)
                or _isolated_squares(board, chess.BLACK)
            ):
                continue
            if kind == "islands" and max(
                _pawn_islands(board, chess.WHITE),
                _pawn_islands(board, chess.BLACK),
            ) < 2:
                continue
            dest = chess.square_name(info["first"].to_square)
            titles = {
                "doubled": f"Сдвоенные пешки: ход на {dest}",
                "passed": f"Проходная пешка: ход на {dest}",
                "isolated": f"Изолированная пешка: ход на {dest}",
                "islands": f"Пешечные островки: ход на {dest}",
            }
            tasks.append(
                _move(
                    titles[kind],
                    "Белые начинают. Найдите лучший ход пешкой и улучшите структуру.",
                    fen,
                    chess.square_name(info["first"].from_square),
                    _explanation(board, info, notes[kind]),
                    [info["first"].uci()],
                    difficulty=2,
                )
            )
            kinds.remove(kind)
            used_fens.add(fen)
            used_ids.add(pid)
            break
    return tasks


def opening_theory_tasks(used_fens) -> dict[str, list[dict]]:
    """Задания по теории дебютов из openings.json (lichess-org/chess-openings)."""
    data = json.loads((ROOT / "backend" / "knowledge" / "openings.json").read_text(encoding="utf-8"))
    positions = []
    for entry in data["openings"]:
        board = chess.Board()
        ok = True
        for uci in entry["moves"][:18]:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                ok = False
                break
            if move not in board.legal_moves:
                ok = False
                break
            positions.append((entry, board.copy(), move))
            board.push(move)
        if not ok:
            continue

    def pick_for(predicate, count):
        chosen = []
        seen: set[str] = set()
        for entry, board, move in positions:
            if len(chosen) >= count:
                break
            if board.turn != chess.WHITE:
                continue
            fen = board.fen()
            if fen in used_fens or fen in seen:
                continue
            piece = board.piece_at(move.from_square)
            if piece is None or piece.color != chess.WHITE:
                continue
            after = board.copy()
            after.push(move)
            san = board.san(move)
            if not predicate(board, move, piece, after, san):
                continue
            seen.add(fen)
            chosen.append((board, move, san, after, piece))
        return chosen

    def make(tasks, board, move, san, after, piece, title, instruction, reason):
        accepted = move.uci()
        tasks.append(
            _move(
                title,
                instruction,
                board.fen(),
                chess.square_name(move.from_square),
                f"Теория рекомендует {san}: {reason}",
                [accepted],
                difficulty=1 if board.fullmove_number <= 4 else 2,
            )
        )
        used_fens.add(board.fen())

    result: dict[str, list[dict]] = {}

    tasks: list[dict] = []
    for board, move, san, after, piece in pick_for(
        lambda b, m, p, a, s: p.piece_type in (chess.KNIGHT, chess.BISHOP)
        and chess.square_rank(m.from_square) == 0
        and not b.is_capture(m),
        8,
    ):
        make(
            tasks,
            board,
            move,
            san,
            after,
            piece,
            f"Развитие: {chess.square_name(move.to_square)}",
            "Сделайте развивающий ход фигурой из начального положения.",
            "фигура покидает первую горизонталь и вступает в игру.",
        )
    result["opening-principles"] = tasks

    tasks = []
    for board, move, san, after, piece in pick_for(
        lambda b, m, p, a, s: p.piece_type in (chess.KNIGHT, chess.BISHOP)
        and (
            a.is_check()
            or any(
                target.color == chess.BLACK and VALUES[target.piece_type] >= 3
                for square in a.attacks(m.to_square)
                if (target := a.piece_at(square)) is not None
            )
        ),
        8,
    ):
        make(
            tasks,
            board,
            move,
            san,
            after,
            piece,
            f"Развитие с темпом: {chess.square_name(move.to_square)}",
            "Развейте фигуру так, чтобы одновременно создать угрозу.",
            "фигура развивается и сразу создаёт угрозу - это выигрыш темпа.",
        )
    result["development-tempo"] = tasks

    tasks = []
    for board, move, san, after, piece in pick_for(
        lambda b, m, p, a, s: p.piece_type == chess.KING and abs(m.to_square - m.from_square) == 2,
        8,
    ):
        make(
            tasks,
            board,
            move,
            san,
            after,
            piece,
            "Рокировка в дебюте",
            "Уберите короля из центра рокировкой.",
            "рокировка завершает развитие и делает короля безопаснее.",
        )
    result["opening-king-safety"] = tasks

    tasks = []
    for board, move, san, after, piece in pick_for(
        lambda b, m, p, a, s: p.piece_type == chess.PAWN
        and chess.square_name(m.to_square) in {"d4", "e4", "c4", "d5", "e5"},
        8,
    ):
        make(
            tasks,
            board,
            move,
            san,
            after,
            piece,
            f"Центр: {chess.square_name(move.to_square)}",
            "Захватите центр пешкой.",
            "пешка занимает центральное поле и открывает пространство фигурам.",
        )
    result["opening-center"] = tasks
    return result


def opening_mistake_tasks(pool, used_fens, used_ids) -> list[dict]:
    spec = {
        "count": 10,
        "themes": ("opening",),
        "rating": (600, 1800),
        "target": 1100,
        "motif": "Ошибка в дебюте",
        "instruction": "Соперник неточно разыграл дебют. Найдите лучший ход.",
    }
    tasks = []
    for puzzle, info in pick(pool, spec, used_fens, used_ids):
        board = info["board"]
        piece = info["piece"]
        tasks.append(
            _move(
                f"Наказание за ошибку: {PIECE_INSTR[piece.piece_type]} на {chess.square_name(info['first'].to_square)}",
                spec["instruction"],
                puzzle["fen"],
                chess.square_name(info["first"].from_square),
                _explanation(board, info, "В дебюте нельзя терять время: соперник сразу наказывает за неточность."),
                [info["first"].uci()],
                difficulty=2,
            )
        )
    return tasks


def discovered_check_tasks() -> list[dict]:
    """Учебные позиции на вскрытый шах (проверено python-chess)."""
    raw = [
        (
            "4k3/5ppp/8/8/8/8/4B3/4R2K w - - 0 1",
            "e2c4",
            "Вскрытый шах: слон уходит с линии",
            "Уйдите слоном с линии e, чтобы открыть шах ладье.",
            "Bc4 освобождает вертикаль e: ладья e1 объявляет шах королю e8, а сам слон короля не атакует.",
        ),
        (
            "1k6/5ppp/8/8/8/8/1B6/1R4K1 w - - 0 1",
            "b2g7",
            "Вскрытый шах: слон открывает ладью",
            "Уйдите слоном с вертикали b, чтобы открыть шах ладье.",
            "Bg7 освобождает вертикаль b: ладья b1 объявляет шах королю b8, а слон на g7 короля не атакует.",
        ),
        (
            "7k/8/8/8/8/8/7N/6KR w - - 0 1",
            "h2g4",
            "Вскрытый шах: конь открывает ладью",
            "Уйдите конём с вертикали h, чтобы открыть шах ладье.",
            "Ng4 освобождает вертикаль h: ладья h1 объявляет шах королю h8, а конь короля не атакует.",
        ),
    ]
    tasks = []
    for fen, uci, title, instruction, explanation in raw:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        after = board.copy()
        after.push(move)
        if not after.is_check():
            continue
        king = after.king(chess.BLACK)
        if king is None or king in after.attacks(move.to_square):
            continue
        tasks.append(
            _move(
                title,
                instruction,
                fen,
                chess.square_name(move.from_square),
                explanation,
                [uci],
                difficulty=2,
            )
        )
    return tasks


def main() -> None:
    pool = load_pool()
    used_fens: set[str] = set()
    used_ids: set[str] = set()

    # Позиции, которые уже есть в курсе, не дублируем.
    os.environ["TRAINING_COURSE_BASE_ONLY"] = "1"
    from backend.services.training_course_data import LESSONS_BY_MODULE  # noqa: WPS433

    for lessons in LESSONS_BY_MODULE.values():
        for lesson in lessons:
            for task in lesson["tasks"]:
                used_fens.add(task["fen"])

    tasks: dict[str, list[dict]] = {}
    tasks.update(basic_tasks(pool, used_fens))
    tasks.update(select_tasks(pool, used_fens))
    tasks.update(special_tasks(pool, used_fens))
    tasks["stalemate"] = stalemate_tasks()
    tasks.update(king_mate_tasks())
    tasks["opposition"] = opposition_tasks()
    tasks["pawn-structure"] = pawn_structure_tasks(pool, used_fens, used_ids)
    tasks.update(opening_theory_tasks(used_fens))
    tasks["opening-mistakes"] = opening_mistake_tasks(pool, used_fens, used_ids)
    tasks.update(_puzzle_tasks(pool, used_fens, used_ids))
    tasks.setdefault("discovered-check", []).extend(discovered_check_tasks())

    # В каждом уроке оставляем 2-3 текстовых вопроса, остальное - позиции.
    for slug, items in tasks.items():
        items.sort(key=lambda task: (0 if task["task_type"] == "choose_option" else 1, task["difficulty"]))
        tasks[slug] = items

    lines = [
        '"""Дополнительные задания курса: реальные позиции из партий.',
        "",
        "Файл сгенерирован scripts/build_course_extra.py. Не редактируйте вручную:",
        "перезапустите генератор. Источники - Lichess (CC0) и chess-openings.",
        '"""',
        "",
        "EXTRA_TASKS_BY_LESSON: dict[str, list[dict]] = {",
    ]
    for slug, items in tasks.items():
        lines.append(f"    {slug!r}: [")
        for task in items:
            lines.append("        {")
            for key in (
                "task_type",
                "title",
                "instruction",
                "fen",
                "source_square",
                "payload",
                "explanation",
                "difficulty",
            ):
                value = task.get(key)
                if key == "payload":
                    lines.append("            'payload': {")
                    for pkey, pvalue in value.items():
                        lines.append(f"                {pkey!r}: {pvalue!r},")
                    lines.append("            },")
                else:
                    lines.append(f"            {key!r}: {value!r},")
            lines.append("        },")
        lines.append("    ],")
    lines.append("}")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    total = sum(len(items) for items in tasks.values())
    print(f"Записано {total} заданий в {OUT}")


if __name__ == "__main__":
    main()
