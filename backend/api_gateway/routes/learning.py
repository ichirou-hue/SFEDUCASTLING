"""Обучающие endpoint'ы: тест определения уровня, проверка тактических задач.

Задача 2.1 «Определение уровня» по ТЗ: тест из N задач возрастающей
сложности (по 5 на каждый Elo-диапазон). Оценка пользователя ставится
по максимальному диапазону, в котором он решает большинство задач.

Проверка ответа гибридная:
- совпадение с эталонным решением (из базы паззлов) — засчитываем;
- иначе — спрашиваем Stockfish: если ход почти так же хорош, как лучший,
  то засчитываем частично (это лояльно к сильным, но не точным ответам).
"""

import random
from datetime import UTC, datetime

import chess
import chess.engine
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway import state
from backend.api_gateway.dependecies import get_current_user, get_optional_current_user
from backend.api_gateway.models import PuzzleAnswerRequest
from backend.db.session import get_db
from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.user import User
from backend.services.adaptive_logic import select_adaptive_puzzles
from backend.services.adaptive_training import build_weakness_profile
from backend.services.dynamic_difficulty import (
    get_user_theme_difficulties,
    public_difficulty_profile,
    update_theme_difficulty_after_attempt,
)
from backend.services.course_topics import (
    COURSE_TOPIC_PUZZLE_THEMES,
    primary_course_topic_for_puzzle,
    puzzle_matches_course_topic as _puzzle_matches_course_topic,
)

router = APIRouter(tags=["learning"])


# Диапазоны Elo по ТЗ (0-500 / 500-1000 / 1000-1500 / 1500+).
# Соответствие с корзинами рейтинга паззлов.
LEVEL_BUCKETS = [
    {
        "level": 1,
        "name": "Новичок",
        "band": 500,
        "min_puzzle_rating": 500,
        "max_puzzle_rating": 900,
    },
    {
        "level": 2,
        "name": "Любитель",
        "band": 1000,
        "min_puzzle_rating": 900,
        "max_puzzle_rating": 1300,
    },
    {
        "level": 3,
        "name": "Клубный",
        "band": 1500,
        "min_puzzle_rating": 1300,
        "max_puzzle_rating": 1700,
    },
    {
        "level": 4,
        "name": "Продвинутый",
        "band": 2000,
        "min_puzzle_rating": 1700,
        "max_puzzle_rating": 9999,
    },
]


class LevelTestSubmitAnswer(BaseModel):
    puzzle_id: str = Field(min_length=1, max_length=64)
    correct: bool


class LevelTestSubmitRequest(BaseModel):
    test_id: int = Field(gt=0)
    answers: list[LevelTestSubmitAnswer] = Field(min_length=1, max_length=100)


class PuzzleAttemptRequest(BaseModel):
    puzzle_id: str = Field(min_length=1, max_length=64)
    correct: bool


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
    """Собирает рандомизированный тестовый набор по Elo-диапазонам.

    Из каждого диапазона случайно выбирается до 5 задач. После этого
    итоговый список перемешивается, чтобы пользователь не получал задачи
    в фиксированном порядке от простых к сложным.
    """
    if not state.puzzle_base:
        return []

    puzzles = state.puzzle_base.get("puzzles", [])
    picked: list[dict] = []

    for bucket in LEVEL_BUCKETS:
        lo, hi = bucket["min_puzzle_rating"], bucket["max_puzzle_rating"]
        pool = [p for p in puzzles if lo <= p.get("rating", 0) < hi]

        # random.sample не изменяет исходный pool и гарантирует отсутствие
        # повторов внутри выборки. min(...) сохраняет старое поведение, если
        # в конкретном диапазоне оказалось меньше пяти задач.
        sample_size = min(5, len(pool))
        if sample_size:
            picked.extend(random.sample(pool, sample_size))

    # Перемешиваем задачи разных диапазонов между собой. Расчёт результата
    # от порядка не зависит: level_test_result определяет диапазон по rating.
    random.shuffle(picked)
    return picked


@router.post("/api/learning/level-test/start")
async def level_test_start(
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_current_user),
):
    """Начинает тест определения уровня.

    Для авторизованного пользователя создаёт строку level_tests и возвращает
    test_id. Для гостя сохраняется прежнее поведение: задачи выдаются, но
    test_id=None и записать результат в профиль нельзя.
    """
    puzzles = _level_test_puzzles()
    if not puzzles:
        return {"error": "База паззлов не загружена", "questions": [], "test_id": None}

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

    test_id: int | None = None
    if user is not None:
        test = LevelTest(
            user_id=user.id,
            question_ids=[q["id"] for q in questions],
            status="started",
        )
        db.add(test)
        await db.commit()
        await db.refresh(test)
        test_id = test.id

    return {"test_id": test_id, "questions": questions, "total": len(questions)}


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


def _score_level_test(answers: list[dict]) -> dict:
    """Считает итог level-test единообразно для /result и /submit."""
    if not state.puzzle_base:
        raise ValueError("База паззлов не загружена")

    by_id = {p.get("id"): p for p in state.puzzle_base.get("puzzles", [])}

    scored = {
        bucket["level"]: {
            "total": 0,
            "correct": 0,
            "name": bucket["name"],
        }
        for bucket in LEVEL_BUCKETS
    }

    for answer in answers:
        pid = answer.get("puzzle_id")
        puzzle = by_id.get(pid)
        if not puzzle:
            continue

        rating = puzzle.get("rating", 0)
        for bucket in LEVEL_BUCKETS:
            lo = bucket["min_puzzle_rating"]
            hi = bucket["max_puzzle_rating"]
            if lo <= rating < hi:
                scored[bucket["level"]]["total"] += 1
                if answer.get("correct"):
                    scored[bucket["level"]]["correct"] += 1
                break

    level = 1
    for bucket in LEVEL_BUCKETS:
        bucket_score = scored[bucket["level"]]
        if bucket_score["total"] >= 3 and bucket_score["correct"] >= 3:
            level = bucket["level"]

    result_bucket = next(b for b in LEVEL_BUCKETS if b["level"] == level)
    return {
        "level": level,
        "band": result_bucket["band"],
        "score": scored,
        "result": {
            "level": level,
            "name": result_bucket["name"],
            "band": result_bucket["band"],
        },
    }


@router.post("/api/learning/level-test/result")
def level_test_result(answers: list[dict]):
    """Считает уровень без изменения профиля (legacy/read-only endpoint)."""
    try:
        return _score_level_test(answers)
    except ValueError as exc:
        return {"error": str(exc)}


@router.post("/api/learning/level-test/submit")
async def level_test_submit(
    req: LevelTestSubmitRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Фиксирует итог теста и записывает его Elo-band в профиль.

    Идемпотентность обеспечивается строкой level_tests: она блокируется
    SELECT ... FOR UPDATE. Если этот test_id уже был завершён, повторный
    submit возвращает сохранённый результат и не пересчитывает его.
    """
    test = await db.scalar(
        select(LevelTest)
        .where(LevelTest.id == req.test_id, LevelTest.user_id == user.id)
        .with_for_update()
    )
    if test is None:
        raise HTTPException(status_code=404, detail="Тест не найден")

    if test.status == "submitted":
        return {
            "ok": True,
            "test_id": test.id,
            "level": test.level,
            "band": test.band,
            "score": test.score,
            "already_submitted": True,
        }

    answers = [answer.model_dump() for answer in req.answers]
    answer_ids = [answer["puzzle_id"] for answer in answers]

    if len(answer_ids) != len(set(answer_ids)):
        raise HTTPException(status_code=422, detail="В ответах есть повторяющиеся puzzle_id")

    expected_ids = list(test.question_ids or [])
    if set(answer_ids) != set(expected_ids):
        raise HTTPException(
            status_code=422,
            detail="Набор ответов не совпадает с задачами этого теста",
        )

    try:
        result = _score_level_test(answers)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Блокируем и пользователя: два разных теста, отправленные одновременно,
    # не должны потерять обновление профиля из-за гонки транзакций.
    locked_user = await db.scalar(
        select(User).where(User.id == user.id).with_for_update()
    )
    if locked_user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    locked_user.elo = result["band"]

    test.answers = answers
    test.score = result["score"]
    test.level = result["level"]
    test.band = result["band"]
    test.status = "submitted"
    test.submitted_at = datetime.now(UTC)

    await db.commit()

    return {
        "ok": True,
        "test_id": test.id,
        "level": test.level,
        "band": test.band,
        "score": test.score,
        "already_submitted": False,
    }


@router.post("/api/learning/puzzle/attempt")
async def record_puzzle_attempt(
    req: PuzzleAttemptRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Сохраняет попытку обычного тактического пазла для прогресса.

    Тренировочная страница сейчас воспроизводит полное решение на клиенте,
    поэтому A4 фиксирует уже известный клиенту итог попытки. puzzle_id при
    этом обязательно проверяется по загруженной базе задач.
    """
    if not state.puzzle_base:
        raise HTTPException(status_code=503, detail="База паззлов не загружена")

    puzzle_lookup = {
        str(p.get("id")): p for p in state.puzzle_base.get("puzzles", [])
    }
    puzzle = puzzle_lookup.get(req.puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    # До сохранения новой попытки лениво инициализируем B2 по уже накопленной
    # истории. Так первая новая попытка меняет difficulty ровно на один шаг.
    await get_user_theme_difficulties(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
    )

    attempt = PuzzleAttempt(
        user_id=user.id,
        puzzle_id=req.puzzle_id,
        correct=req.correct,
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)

    topic_slug = primary_course_topic_for_puzzle(puzzle)
    difficulty_update = await update_theme_difficulty_after_attempt(
        db,
        user_id=user.id,
        theme_slug=topic_slug,
        puzzle_base=state.puzzle_base,
    )

    return {
        "ok": True,
        "attempt_id": attempt.id,
        "puzzle_id": attempt.puzzle_id,
        "correct": attempt.correct,
        "topic": topic_slug,
        "difficulty_update": difficulty_update,
    }


@router.get("/api/learning/progress")
async def learning_progress(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Прогресс текущего пользователя по обычным тактическим пазлам."""
    total_attempts = int(
        await db.scalar(
            select(func.count(PuzzleAttempt.id)).where(PuzzleAttempt.user_id == user.id)
        )
        or 0
    )
    correct_attempts = int(
        await db.scalar(
            select(func.count(PuzzleAttempt.id)).where(
                PuzzleAttempt.user_id == user.id,
                PuzzleAttempt.correct.is_(True),
            )
        )
        or 0
    )
    attempted = int(
        await db.scalar(
            select(func.count(func.distinct(PuzzleAttempt.puzzle_id))).where(
                PuzzleAttempt.user_id == user.id
            )
        )
        or 0
    )
    solved = int(
        await db.scalar(
            select(func.count(func.distinct(PuzzleAttempt.puzzle_id))).where(
                PuzzleAttempt.user_id == user.id,
                PuzzleAttempt.correct.is_(True),
            )
        )
        or 0
    )

    accuracy = round(correct_attempts * 100 / total_attempts, 1) if total_attempts else None
    return {
        "attempted": attempted,
        "solved": solved,
        "attempts": total_attempts,
        "correct_attempts": correct_attempts,
        "accuracy": accuracy,
    }


@router.get("/api/learning/difficulty")
async def learning_difficulty(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Текущая B2-сложность пользователя по каждой учебной теме."""
    if not state.puzzle_base:
        raise HTTPException(status_code=503, detail="База паззлов не загружена")

    profile, difficulties = await get_user_theme_difficulties(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
    )
    return {
        "topics": public_difficulty_profile(profile, difficulties),
        "rule": {
            "min_difficulty": 1,
            "max_difficulty": 3,
            "default_difficulty": 2,
            "increase_if_accuracy_gt": 80,
            "decrease_if_accuracy_lt": 50,
        },
    }


@router.get("/api/learning/weaknesses")
async def learning_weaknesses(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Профиль сильных/слабых тем по объединённым учебным попыткам."""
    if not state.puzzle_base:
        raise HTTPException(status_code=503, detail="База паззлов не загружена")
    return await build_weakness_profile(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
    )


@router.get("/api/learning/puzzles/adaptive")
async def get_adaptive_puzzles(
    count: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Персональный набор: 70% слабые темы, 30% сильные темы."""
    if not state.puzzle_base:
        raise HTTPException(status_code=503, detail="База паззлов не загружена")

    profile, topic_difficulties = await get_user_theme_difficulties(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
    )
    valid_puzzles = [
        p for p in state.puzzle_base.get("puzzles", []) if _is_puzzle_valid(p)
    ]
    picked, allocation = select_adaptive_puzzles(
        valid_puzzles,
        weak_topics=profile["weak_topics"],
        strong_topics=profile["strong_topics"],
        count=count,
        topic_difficulties=topic_difficulties,
    )

    result = [
        {
            "id": p.get("id", ""),
            "fen": p.get("fen", ""),
            "moves": p.get("moves", ""),
            "themes": p.get("themes", []),
            "rating": p.get("rating", 0),
            "adaptive_group": p.get("adaptive_group"),
            "adaptive_topic": p.get("adaptive_topic"),
            "adaptive_difficulty": p.get("adaptive_difficulty"),
            "difficulty_match": p.get("difficulty_match"),
        }
        for p in picked
    ]

    return {
        "puzzles": result,
        "total": len(result),
        "mode": "adaptive",
        "weak_topics": profile["weak_topics"],
        "strong_topics": profile["strong_topics"],
        "difficulty_profile": public_difficulty_profile(profile, topic_difficulties),
        "allocation": allocation,
        "selection_rule": profile["selection_rule"],
    }


@router.post("/api/learning/puzzle/check")
def puzzle_check(req: PuzzleAnswerRequest):
    """Проверка решения отдельного паззла (для миттельшпиля, Задача 2.4)."""
    return level_test_check(req)


@router.get("/api/learning/puzzles")
def get_puzzles(count: int = 20, topic: str | None = None):
    """Возвращает набор задач с полными решениями для тренировочной страницы.

    ``topic`` — slug модуля учебного курса. Если он передан, выдаются только
    связанные с этим модулем Lichess-темы. Так TrainingPage может отправить
    пользователя сразу на релевантную практику.

    Отличие от level-test/start: включает поле moves (ходы решения).
    Задачи, где после решения игрок оказывается матован, исключаются.
    """
    if not state.puzzle_base:
        return {"error": "База паззлов не загружена", "puzzles": []}

    normalized_topic = (topic or "").strip() or None
    if normalized_topic and normalized_topic not in COURSE_TOPIC_PUZZLE_THEMES:
        raise HTTPException(status_code=400, detail="Неизвестная тема учебного курса")

    all_puzzles = state.puzzle_base.get("puzzles", [])

    valid_puzzles = [p for p in all_puzzles if _is_puzzle_valid(p)]
    if normalized_topic:
        valid_puzzles = [
            p for p in valid_puzzles if _puzzle_matches_course_topic(p, normalized_topic)
        ]

    picked: list[dict] = []
    selected_ids: set[str] = set()
    per_bucket = max(1, count // len(LEVEL_BUCKETS)) if count > 0 else 0

    for bucket in LEVEL_BUCKETS:
        lo, hi = bucket["min_puzzle_rating"], bucket["max_puzzle_rating"]
        pool = [p for p in valid_puzzles if lo <= p.get("rating", 0) < hi]
        random.shuffle(pool)
        for puzzle in pool[:per_bucket]:
            picked.append(puzzle)
            selected_ids.add(str(puzzle.get("id", "")))

    # Тематическая выборка может быть неравномерной по Elo. Добираем оставшиеся
    # задачи из той же темы, чтобы ссылка «проверить на пазлах» по возможности
    # всё равно давала полноценный набор из count позиций.
    if len(picked) < count:
        remaining = [
            p for p in valid_puzzles if str(p.get("id", "")) not in selected_ids
        ]
        random.shuffle(remaining)
        picked.extend(remaining[: max(0, count - len(picked))])

    random.shuffle(picked)

    result = []
    for p in picked[:count]:
        result.append({
            "id": p.get("id", ""),
            "fen": p.get("fen", ""),
            "moves": p.get("moves", ""),
            "themes": p.get("themes", []),
            "rating": p.get("rating", 0),
        })

    return {
        "puzzles": result,
        "total": len(result),
        "topic": normalized_topic,
    }


def _is_puzzle_valid(puzzle: dict) -> bool:
    """Проверяет, что задача валидна для тренировки.

    Исключает задачи, где после решения:
    - игрок матован;
    - игрок остаётся под шахом (позиция не стабилизировалась).
    """
    try:
        board = chess.Board(puzzle["fen"])
        moves_str = puzzle.get("moves", "").strip()
        if not moves_str:
            return False
        for uci in moves_str.split():
            board.push(chess.Move.from_uci(uci))

        initial_turn = puzzle["fen"].split()[1]
        player_color = chess.WHITE if initial_turn == "w" else chess.BLACK

        if board.is_checkmate():
            mated = board.turn == player_color
            return not mated

        if board.is_check() and board.turn == player_color:
            return False

        return True
    except Exception:
        return False
