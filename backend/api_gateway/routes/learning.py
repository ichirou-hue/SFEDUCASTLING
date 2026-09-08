"""Обучающие endpoint'ы: тест определения уровня, проверка тактических задач.

Задача 2.1 «Определение уровня» по ТЗ: тест из N задач возрастающей
сложности (по 5 на каждый Elo-диапазон). Оценка пользователя ставится
по максимальному диапазону, в котором он решает большинство задач.

Проверка ответа гибридная:
- совпадение с эталонным решением (из базы паззлов) — засчитываем;
- иначе — спрашиваем Stockfish: если ход почти так же хорош, как лучший,
  то засчитываем частично (это лояльно к сильным, но не точным ответам).
"""

import chess
import chess.engine
from fastapi import APIRouter

from backend.api_gateway import state
from backend.api_gateway.models import PuzzleAnswerRequest

router = APIRouter(tags=["learning"])


# Диапазоны Elo по ТЗ (0-500 / 500-1000 / 1000-1500 / 1500+).
# Соответствие с корзинами рейтинга паззлов.
LEVEL_BUCKETS = [
    {"level": 1, "name": "Новичок",       "min_puzzle_rating": 500,  "max_puzzle_rating": 900},
    {"level": 2, "name": "Любитель",      "min_puzzle_rating": 900,  "max_puzzle_rating": 1300},
    {"level": 3, "name": "Клубный",       "min_puzzle_rating": 1300, "max_puzzle_rating": 1700},
    {"level": 4, "name": "Продвинутый",   "min_puzzle_rating": 1700, "max_puzzle_rating": 9999},
]


def _ordinal_to_uci(board: chess.Board, san: str) -> str | None:
    """Переводит SAN-ход (например 'Nxe5') в UCI для текущей позиции."""
    try:
        move = board.parse_san(san)
        if move in board.legal_moves:
            return move.uci()
    except Exception:
        return None
    return None


def _solution_for(board: chess.Board, moves_str: str) -> str | None:
    """Возвращает UCI первого хода решения из строки Moves (все в UCI)."""
    parts = [p for p in moves_str.split() if p]
    if not parts:
        return None
    return parts[0]


def _level_test_puzzles() -> list[dict]:
    """Собирает тестовый набор: по 5 задач на каждый Elo-диапазон."""
    if not state.puzzle_base:
        return []
    puzzles = state.puzzle_base.get("puzzles", [])
    picked: list[dict] = []
    for bucket in LEVEL_BUCKETS:
        lo, hi = bucket["min_puzzle_rating"], bucket["max_puzzle_rating"]
        pool = [p for p in puzzles if lo <= p.get("rating", 0) < hi]
        pool = sorted(pool, key=lambda p: p.get("rating", 0))
        picked.extend(pool[:5])
    return picked


@router.post("/api/learning/level-test/start")
def level_test_start():
    """Начинает тест определения уровня.

    Возвращает набор задач (без решений) сгруппированных по сложности.
    """
    puzzles = _level_test_puzzles()
    if not puzzles:
        return {"error": "База паззлов не загружена", "questions": []}

    questions = []
    for p in puzzles:
        questions.append(
            {
                "id": p.get("id", ""),
                "fen": p.get("fen", ""),
                "themes": p.get("themes", []),
                "rating": p.get("rating", 0),
            }
        )
    return {"questions": questions, "total": len(questions)}


@router.post("/api/learning/level-test/check")
def level_test_check(req: PuzzleAnswerRequest):
    """Проверяет ответ на задачу теста уровня.

    Возвращает: correct (bool), solution, explanation.
    """
    if not state.puzzle_base:
        return {"error": "База паззлов не загружена", "correct": False, "solution": None}

    puzzles = state.puzzle_base.get("puzzles", [])
    puzzle = next((p for p in puzzles if p.get("id") == req.puzzle_id), None)
    if puzzle is None:
        return {"error": "Задача не найдена", "correct": False, "solution": None}

    try:
        board = chess.Board(puzzle["fen"])
    except Exception:
        return {"error": "Некорректная позиция в задаче", "correct": False, "solution": None}

    solution = _solution_for(board, puzzle.get("moves", ""))
    if solution is None:
        return {"error": "У задачи нет решения", "correct": False, "solution": None}

    # Пытаемся принять SAN, если пользователь прислал не UCI.
    user_move = req.move.strip()
    if len(user_move) not in (4, 5):
        uci_san = _ordinal_to_uci(board, user_move)
        if uci_san is None:
            return {"correct": False, "solution": solution, "message": "Некорректный ход"}
        user_move = uci_san

    correct = user_move == solution

    result = {
        "correct": correct,
        "solution": solution,
        "puzzle_rating": puzzle.get("rating", 0),
        "themes": puzzle.get("themes", []),
    }
    if not correct:
        # Гибкая проверка через Stockfish: сильный ход засчитываем частично.
        try:
            move_ok, is_best = _stockfish_matches(board, user_move, solution)
            result["strong_but_different"] = move_ok
            result["is_best"] = is_best
        except Exception:
            pass
    return result


def _stockfish_matches(board: chess.Board, user_move: str, solution: str) -> tuple[bool, bool]:
    """Выясняет, насколько ход пользователя близок к лучшему по Stockfish.

    Возвращает (ходит_почти_так_же_хорошо, это_лучший_ход).
    """
    engine = state.ensure_stockfish()
    if engine is None:
        return False, False

    try:
        import stockfish as sf_module  # только для типа
    except Exception:
        pass

    engine.set_fen_position(board.fen())
    try:
        best = engine.get_best_move()
    except Exception:
        return False, False

    is_best = (user_move == best)

    # Сравниваем оценки хода пользователя и лучшего хода.
    def _eval_uci(uci: str) -> float | None:
        try:
            m = chess.Move.from_uci(uci)
            if m not in board.legal_moves:
                return None
            board.push(m)
            engine.set_fen_position(board.fen())
            try:
                info = engine.get_evaluation()
            finally:
                board.pop()
            if info.get("type") == "cp":
                return info["value"]
            if info.get("type") == "mate":
                return 10000 if info["value"] > 0 else -10000
        except Exception:
            return None
        return None

    user_eval = _eval_uci(user_move)
    best_eval = _eval_uci(best) if best else None
    if user_eval is None or best_eval is None:
        return False, False

    # Если разница не больше 50 центипешек — ход почти так же хорош.
    return abs(user_eval - best_eval) <= 50, is_best


@router.post("/api/learning/level-test/result")
def level_test_result(answers: list[dict]):
    """Считает уровень по списку ответов.

    answers: [{"puzzle_id": ..., "correct": bool}, ...]

    Уровень = наибольший диапазон, в котором решено >=3 задач из 5.
    """
    if not state.puzzle_base:
        return {"error": "База паззлов не загружена"}

    by_id = {p.get("id"): p for p in state.puzzle_base.get("puzzles", [])}

    scored = {level["level"]: {"total": 0, "correct": 0} for level in LEVEL_BUCKETS}
    for bucket in LEVEL_BUCKETS:
        lo, hi = bucket["min_puzzle_rating"], bucket["max_puzzle_rating"]
        scored[bucket["level"]] = {"total": 0, "correct": 0, "name": bucket["name"]}

    for a in answers:
        pid = a.get("puzzle_id")
        p = by_id.get(pid)
        if not p:
            continue
        rating = p.get("rating", 0)
        for bucket in LEVEL_BUCKETS:
            lo, hi = bucket["min_puzzle_rating"], bucket["max_puzzle_rating"]
            if lo <= rating < hi:
                scored[bucket["level"]]["total"] += 1
                if a.get("correct"):
                    scored[bucket["level"]]["correct"] += 1
                break

    level = 1
    for bucket in LEVEL_BUCKETS:
        s = scored[bucket["level"]]
        if s["total"] >= 3 and s["correct"] >= 3:
            level = bucket["level"]

    return {
        "level": level,
        "score": scored,
        "result": {
            "level": level,
            "name": next(b["name"] for b in LEVEL_BUCKETS if b["level"] == level),
        },
    }


@router.post("/api/learning/puzzle/check")
def puzzle_check(req: PuzzleAnswerRequest):
    """Проверка решения отдельного паззла (для миттельшпиля, Задача 2.4)."""
    return level_test_check(req)
