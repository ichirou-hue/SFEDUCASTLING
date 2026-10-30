"""Обучение, стартовая оценка и адаптивные тактические задания."""

from __future__ import annotations

import random
import secrets
from datetime import UTC, datetime
from typing import Any

import chess
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway import state
from backend.api_gateway.dependecies import get_current_user, get_optional_current_user
from backend.api_gateway.models import PuzzleAnswerRequest
from backend.db.session import get_db
from backend.models.game import Game
from backend.models.gigacoin_event import GigacoinEvent
from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.training_attempt import TrainingAttempt
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask
from backend.models.user import ROLE_ADMIN, User
from backend.models.user_chess_account import UserChessAccount
from backend.services.adaptive_logic import select_adaptive_puzzles
from backend.services.adaptive_training import (
    build_weakness_profile,
    compute_learning_progress,
)
from backend.services.assessment import (
    RATING_GROUPS,
    build_test_feedback,
    calculate_performance_rating,
    max_puzzle_rating_for_user,
    personalized_level_test_puzzles,
    puzzle_rating_group,
    rating_band_progress,
    rating_group_for,
    rating_for_selection,
)
from backend.services.course_topics import (
    COURSE_TOPIC_PUZZLE_THEMES,
    primary_course_topic_for_puzzle,
)
from backend.services.course_topics import (
    puzzle_matches_course_topic as _puzzle_matches_course_topic,
)
from backend.services.dynamic_difficulty import (
    effective_player_rating,
    get_user_theme_difficulties,
    public_difficulty_profile,
    update_theme_difficulty_after_attempt,
)
from backend.services.difficulty_logic import difficulty_for_rating
from backend.services.phase_accuracy import load_user_phase_accuracy
from backend.services.user_records import build_user_records
from backend.services.gigacoin_service import (
    LEVEL_TEST_REWARD,
    PUZZLE_STREAK_REWARDS,
    award_gigacoins,
    puzzle_reward,
)

router = APIRouter(tags=["learning"])


# Общие рейтинговые корзины для обычной страницы паззлов. Границы совпадают
# с новой стартовой оценкой: 0–1000 / 1001–1500 / 1501–1900 / 1901+.
LEVEL_BUCKETS = [
    {"min_puzzle_rating": 0, "max_puzzle_rating": 1001},
    {"min_puzzle_rating": 1001, "max_puzzle_rating": 1501},
    {"min_puzzle_rating": 1501, "max_puzzle_rating": 1901},
    {"min_puzzle_rating": 1901, "max_puzzle_rating": 10000},
]

# Пробный доступ до анкеты. Зарегистрированный, но ещё не прошедший анкету
# пользователь может пройти не больше трёх лёгких задач (рейтинг до 700),
# после чего его просят заполнить анкету. Гостей это не касается — у них
# свой гейт на предпоследнем ходу.
PREVIEW_MAX_RATING = 700
PREVIEW_PUZZLE_LIMIT = 3


class LevelTestCheckRequest(BaseModel):
    test_id: int = Field(gt=0)
    puzzle_id: str = Field(min_length=1, max_length=64)
    move: str = Field(min_length=4, max_length=16)
    response_time_ms: int | None = Field(default=None, ge=0, le=3_600_000)


class LevelTestSkipRequest(BaseModel):
    test_id: int = Field(gt=0)
    puzzle_id: str = Field(min_length=1, max_length=64)
    response_time_ms: int | None = Field(default=None, ge=0, le=3_600_000)


class PuzzleAttemptRequest(BaseModel):
    puzzle_id: str = Field(min_length=1, max_length=64)
    correct: bool
    player_moves: list[str] = Field(default_factory=list, max_length=50)


class LevelTestSubmitAnswer(BaseModel):
    puzzle_id: str = Field(min_length=1, max_length=64)
    correct: bool


class LevelTestSubmitRequest(BaseModel):
    test_id: int = Field(gt=0)
    # Оставлено для совместимости со старым клиентом. Новый клиент хранит
    # прогресс на сервере через /check и /skip.
    answers: list[LevelTestSubmitAnswer] = Field(default_factory=list, max_length=100)


def _puzzle_lookup() -> dict[str, dict[str, Any]]:
    return {
        str(p.get("id")): p
        for p in (state.puzzle_base or {}).get("puzzles", [])
        if p.get("id")
    }


def _public_question(puzzle: dict[str, Any]) -> dict[str, Any]:
    # Не возвращаем rating/themes/moves: они могут подсказать сложность и решение.
    return {
        "id": str(puzzle.get("id", "")),
        "fen": puzzle.get("fen", ""),
    }


def _ordinal_to_uci(board: chess.Board, san: str) -> str | None:
    try:
        move = board.parse_san(san)
        if move in board.legal_moves:
            return move.uci()
    except Exception:
        return None
    return None


def _solution_for(board: chess.Board, moves_str: str) -> str | None:
    parts = [p for p in moves_str.split() if p]
    return parts[0] if parts else None


def _stockfish_matches(board: chess.Board, user_move: str, solution: str) -> tuple[bool, bool]:
    engine = state.ensure_stockfish()
    if engine is None:
        return False, False
    engine.set_fen_position(board.fen())
    try:
        best = engine.get_best_move()
    except Exception:
        return False, False

    is_best = user_move == best

    def _eval_uci(uci: str) -> float | None:
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return None
            board.push(move)
            engine.set_fen_position(board.fen())
            try:
                info = engine.get_evaluation()
            finally:
                board.pop()
            if info.get("type") == "cp":
                return float(info["value"])
            if info.get("type") == "mate":
                return 10000.0 if info["value"] > 0 else -10000.0
        except Exception:
            return None
        return None

    user_eval = _eval_uci(user_move)
    best_eval = _eval_uci(best) if best else None
    if user_eval is None or best_eval is None:
        return False, is_best
    return abs(user_eval - best_eval) <= 50, is_best


async def _latest_personal_test(db: AsyncSession, user_id: int) -> LevelTest | None:
    return await db.scalar(
        select(LevelTest)
        .where(
            LevelTest.user_id == user_id,
            LevelTest.status == "started",
            LevelTest.initial_rating.is_not(None),
        )
        .order_by(LevelTest.id.desc())
        .limit(1)
    )


async def _starting_rating(db: AsyncSession, user: User) -> int:
    onboarding = user.onboarding if isinstance(user.onboarding, dict) else {}
    answers = onboarding.get("answers") if isinstance(onboarding.get("answers"), dict) else {}
    q2 = answers.get("q2") if isinstance(answers.get("q2"), dict) else {}
    requested_platform = str(q2.get("platform") or "").strip().lower()
    if requested_platform in {"chess.com", "chess_com", "chess"}:
        requested_platform = "chesscom"
    if requested_platform not in {"lichess", "chesscom"}:
        requested_platform = ""

    account_query = select(UserChessAccount).where(
        UserChessAccount.user_id == user.id,
        UserChessAccount.rating.is_not(None),
        UserChessAccount.rating_usable.is_(True),
    )
    if requested_platform:
        account_query = account_query.where(
            UserChessAccount.platform == requested_platform
        )
    account = await db.scalar(account_query.order_by(UserChessAccount.updated_at.desc()))
    if account is not None:
        return max(0, min(3500, int(account.rating)))

    return max(0, min(3500, int(user.rating_estimate or user.elo or 800)))


def _upsert_test_answer(test: LevelTest, answer: dict[str, Any]) -> None:
    answers = [
        item
        for item in (test.answers or [])
        if str(item.get("puzzle_id")) != str(answer.get("puzzle_id"))
    ]
    answers.append(answer)
    test.answers = answers


@router.post("/api/learning/level-test/start")
async def level_test_start(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Создаёт или возобновляет персональный 20-задачный level-test."""
    if user.effective_role == ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Администратору стартовая оценка не требуется")
    if not isinstance(user.onboarding, dict) or not user.onboarding.get("submitted_at"):
        raise HTTPException(status_code=409, detail="Сначала заполните анкету Q1–Q8")

    lookup = _puzzle_lookup()
    if not lookup:
        raise HTTPException(status_code=503, detail="База паззлов не загружена")

    active = await _latest_personal_test(db, user.id)
    if active is not None:
        questions = [_public_question(lookup[pid]) for pid in active.question_ids if pid in lookup]
        answered = list(active.answers or [])
        return {
            "test_id": active.id,
            "questions": questions,
            "total": len(questions),
            "answers": answered,
            "resumed": True,
            "personalization": active.metrics_snapshot or {},
        }

    rating_estimate = await _starting_rating(db, user)
    seed = secrets.randbelow(2_147_483_647)
    puzzles, metrics = await personalized_level_test_puzzles(
        db,
        user_id=user.id,
        rating_estimate=rating_estimate,
        puzzle_base=state.puzzle_base,
        seed=seed,
        total=20,
    )
    if not puzzles:
        raise HTTPException(status_code=503, detail="Не удалось сформировать персональный тест")

    group = rating_group_for(rating_estimate)
    test = LevelTest(
        user_id=user.id,
        question_ids=[str(p.get("id")) for p in puzzles],
        answers=[],
        status="started",
        seed=seed,
        initial_rating=rating_estimate,
        rating_group=group["key"],
        metrics_snapshot=metrics,
    )
    db.add(test)
    await db.commit()
    await db.refresh(test)

    return {
        "test_id": test.id,
        "questions": [_public_question(p) for p in puzzles],
        "total": len(puzzles),
        "answers": [],
        "resumed": False,
        "personalization": metrics,
    }


@router.post("/api/learning/level-test/check")
async def level_test_check(
    req: LevelTestCheckRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    test = await db.scalar(
        select(LevelTest).where(
            LevelTest.id == req.test_id,
            LevelTest.user_id == user.id,
        )
    )
    if test is None:
        raise HTTPException(status_code=404, detail="Тест не найден")
    if test.status != "started":
        raise HTTPException(status_code=409, detail="Тест уже завершён")
    if req.puzzle_id not in set(test.question_ids or []):
        raise HTTPException(status_code=422, detail="Эта задача не относится к тесту")

    puzzle = _puzzle_lookup().get(req.puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    try:
        board = chess.Board(puzzle["fen"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Некорректная позиция в задаче") from exc

    solution = _solution_for(board, puzzle.get("moves", ""))
    if solution is None:
        raise HTTPException(status_code=500, detail="У задачи нет решения")

    user_move = req.move.strip()
    if len(user_move) not in (4, 5):
        user_move = _ordinal_to_uci(board, user_move) or ""
    correct = user_move == solution
    strong_but_different = False
    is_best = False
    if not correct and user_move:
        try:
            strong_but_different, is_best = _stockfish_matches(board, user_move, solution)
        except Exception:
            pass

    _upsert_test_answer(
        test,
        {
            "puzzle_id": req.puzzle_id,
            "correct": bool(correct),
            "move": req.move,
            "response_time_ms": req.response_time_ms,
            "skipped": False,
        },
    )
    await db.commit()

    # Эталон намеренно не выдаётся до submit.
    return {
        "correct": bool(correct),
        "strong_but_different": bool(strong_but_different),
        "is_best": bool(is_best),
        "recorded": True,
    }


@router.post("/api/learning/level-test/skip")
async def level_test_skip(
    req: LevelTestSkipRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    test = await db.scalar(
        select(LevelTest).where(LevelTest.id == req.test_id, LevelTest.user_id == user.id)
    )
    if test is None:
        raise HTTPException(status_code=404, detail="Тест не найден")
    if test.status != "started":
        raise HTTPException(status_code=409, detail="Тест уже завершён")
    if req.puzzle_id not in set(test.question_ids or []):
        raise HTTPException(status_code=422, detail="Эта задача не относится к тесту")

    _upsert_test_answer(
        test,
        {
            "puzzle_id": req.puzzle_id,
            "correct": False,
            "move": None,
            "response_time_ms": req.response_time_ms,
            "skipped": True,
        },
    )
    await db.commit()
    return {"ok": True, "recorded": True}


@router.post("/api/learning/level-test/result")
def level_test_result(answers: list[dict]):
    """Legacy preview без записи в профиль."""
    lookup = _puzzle_lookup()
    if not lookup:
        return {"error": "База паззлов не загружена"}
    final_rating = calculate_performance_rating(
        initial_rating=1000, answers=answers, puzzle_lookup=lookup
    )
    group = rating_group_for(final_rating)
    return {
        "level": group["id"],
        "band": group["id"],
        "rating": final_rating,
        "result": {"level": group["id"], "name": group["title"], "band": group["id"], "rating": final_rating},
    }


@router.post("/api/learning/level-test/submit")
async def level_test_submit(
    req: LevelTestSubmitRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    test = await db.scalar(
        select(LevelTest)
        .where(LevelTest.id == req.test_id, LevelTest.user_id == user.id)
        .with_for_update()
    )
    if test is None:
        raise HTTPException(status_code=404, detail="Тест не найден")

    if test.status == "submitted":
        gigacoins_earned = await award_gigacoins(
            db, user, "level_test", "initial-assessment", LEVEL_TEST_REWARD
        )
        if gigacoins_earned:
            await db.commit()
        return {
            "ok": True,
            "test_id": test.id,
            "level": test.level,
            "band": test.band,
            "rating": (test.score or {}).get("final_rating"),
            "score": test.score,
            "result": (test.score or {}).get("result"),
            "feedback": (test.score or {}).get("feedback", []),
            "already_submitted": True,
            "gigacoins_earned": gigacoins_earned,
            "gigacoins_balance": int(user.gigacoins or 0),
        }

    expected_ids = [str(x) for x in (test.question_ids or [])]
    recorded = list(test.answers or [])

    # Старый клиент мог прислать ответы только на submit. Новый сохраняет их
    # серверно после каждого вопроса. При наличии полного серверного прогресса
    # клиентские bool игнорируются.
    recorded_by_id = {str(a.get("puzzle_id")): a for a in recorded}
    if set(recorded_by_id) != set(expected_ids) and req.answers:
        client_answers = [a.model_dump() for a in req.answers]
        if {str(a["puzzle_id"]) for a in client_answers} == set(expected_ids):
            recorded = client_answers
            recorded_by_id = {str(a["puzzle_id"]): a for a in recorded}

    if set(recorded_by_id) != set(expected_ids):
        raise HTTPException(
            status_code=422,
            detail=f"Ответьте на все задачи: сохранено {len(recorded_by_id)} из {len(expected_ids)}",
        )

    answers = [recorded_by_id[pid] for pid in expected_ids]
    lookup = _puzzle_lookup()
    initial_rating = int(test.initial_rating or user.rating_estimate or user.elo or 800)
    final_rating = calculate_performance_rating(
        initial_rating=initial_rating,
        answers=answers,
        puzzle_lookup=lookup,
    )
    group = rating_group_for(final_rating)

    by_group = {
        str(g["id"]): {"name": g["title"], "range": g["key"], "total": 0, "correct": 0}
        for g in RATING_GROUPS
    }
    for answer in answers:
        puzzle = lookup.get(str(answer.get("puzzle_id")))
        if not puzzle:
            continue
        gid = str(puzzle_rating_group(puzzle))
        by_group[gid]["total"] += 1
        if answer.get("correct"):
            by_group[gid]["correct"] += 1

    correct_count = sum(1 for a in answers if a.get("correct"))
    feedback = build_test_feedback(final_rating, answers, lookup)
    score = {
        "total": len(answers),
        "correct": correct_count,
        "accuracy": round(correct_count * 100 / len(answers), 1) if answers else 0,
        "initial_rating": initial_rating,
        "final_rating": final_rating,
        "rating_group": group["key"],
        "by_group": by_group,
        "feedback": feedback,
        "result": {
            "level": int(group["id"]),
            "name": group["title"],
            "band": int(group["id"]),
            "rating": final_rating,
            "rating_group": group["key"],
        },
    }

    locked_user = await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if locked_user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    locked_user.elo = final_rating
    locked_user.skill_band = int(group["id"])
    locked_user.assessment_completed_at = datetime.now(UTC)

    test.answers = answers
    test.score = score
    test.level = int(group["id"])
    test.band = int(group["id"])
    test.status = "submitted"
    test.submitted_at = datetime.now(UTC)
    gigacoins_earned = await award_gigacoins(
        db, locked_user, "level_test", "initial-assessment", LEVEL_TEST_REWARD
    )
    await db.commit()

    return {
        "ok": True,
        "test_id": test.id,
        "level": test.level,
        "band": test.band,
        "rating": final_rating,
        "score": score,
        "result": score["result"],
        "feedback": feedback,
        "already_submitted": False,
        "gigacoins_earned": gigacoins_earned,
        "gigacoins_balance": int(locked_user.gigacoins or 0),
    }


@router.post("/api/learning/puzzle/attempt")
async def record_puzzle_attempt(
    req: PuzzleAttemptRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Сохраняет попытку обычного тактического пазла для прогресса.

    Рейтинг и последовательность ожидаемых ходов берутся из загруженной базы;
    клиент должен передать именно последовательность ходов игрока из решения.
    """
    if not state.puzzle_base:
        raise HTTPException(status_code=503, detail="База паззлов не загружена")
    user = await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    puzzle_lookup = {
        str(p.get("id")): p for p in state.puzzle_base.get("puzzles", [])
    }
    puzzle = puzzle_lookup.get(req.puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if req.correct:
        solution_moves = [move.lower() for move in str(puzzle.get("moves") or "").split()]
        expected_player_moves = solution_moves[::2]
        submitted_player_moves = [str(move).strip().lower() for move in req.player_moves]
        if not expected_player_moves or submitted_player_moves != expected_player_moves:
            raise HTTPException(status_code=422, detail="Ходы не подтверждают решение задачи")

    # До сохранения новой попытки лениво инициализируем B2 по уже накопленной
    # истории. Так первая новая попытка меняет difficulty ровно на один шаг.
    await get_user_theme_difficulties(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
        player_rating=effective_player_rating(user),
    )

    already_solved = await db.scalar(select(PuzzleAttempt.id).where(
        PuzzleAttempt.user_id == user.id,
        PuzzleAttempt.puzzle_id == req.puzzle_id,
        PuzzleAttempt.correct.is_(True),
    ).limit(1)) is not None

    attempt = PuzzleAttempt(
        user_id=user.id,
        puzzle_id=req.puzzle_id,
        correct=req.correct,
    )
    db.add(attempt)
    await db.flush()
    coins_due = puzzle_reward(puzzle.get("rating")) if req.correct and not already_solved else 0
    coins_earned = 0
    base_coins_earned = 0
    if not req.correct:
        user.puzzle_streak = 0
    elif not already_solved:
        user.puzzle_streak = int(user.puzzle_streak or 0) + 1

    if coins_due:
        coins_earned = await award_gigacoins(
            db, user, "puzzle", req.puzzle_id, coins_due
        )
        base_coins_earned = int(await db.scalar(select(GigacoinEvent.amount).where(
            GigacoinEvent.user_id == user.id,
            GigacoinEvent.reason == "puzzle",
            GigacoinEvent.ref == req.puzzle_id,
        )) or 0)
    else:
        # Даже неверная попытка является учебной активностью для дневной серии.
        coins_earned = await award_gigacoins(db, user, "puzzle_activity", str(attempt.id), 0)

    streak_bonus = 0
    streak_reward = PUZZLE_STREAK_REWARDS.get(int(user.puzzle_streak or 0))
    if streak_reward and req.correct and not already_solved:
        streak_bonus = await award_gigacoins(
            db, user, "puzzle_streak", str(attempt.id), streak_reward
        )
        coins_earned += streak_bonus
    await db.commit()
    await db.refresh(attempt)

    topic_slug = primary_course_topic_for_puzzle(puzzle)
    difficulty_update = await update_theme_difficulty_after_attempt(
        db,
        user_id=user.id,
        theme_slug=topic_slug,
        puzzle_base=state.puzzle_base,
        player_rating=effective_player_rating(user),
    )

    return {
        "ok": True,
        "attempt_id": attempt.id,
        "puzzle_id": attempt.puzzle_id,
        "correct": attempt.correct,
        "topic": topic_slug,
        "difficulty_update": difficulty_update,
        "gigacoins_earned": coins_earned,
        "gigacoins_capped": coins_due > base_coins_earned,
        "gigacoins_balance": int(user.gigacoins or 0),
        "already_solved": already_solved,
        "puzzle_streak": int(user.puzzle_streak or 0),
        "puzzle_streak_bonus": streak_bonus,
    }


@router.get("/api/learning/progress")
async def learning_progress(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Прогресс текущего пользователя по обычным тактическим пазлам."""
    return await compute_learning_progress(db, user_id=user.id)


@router.get("/api/learning/records")
async def learning_user_records(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Возвращает личные рекорды и активность за последние 14 дней."""
    games = await db.scalars(
        select(Game)
        .where(Game.user_id == user.id)
        .order_by(Game.created_at, Game.id)
    )
    puzzle_attempts = await db.scalars(
        select(PuzzleAttempt)
        .where(PuzzleAttempt.user_id == user.id)
        .order_by(PuzzleAttempt.created_at, PuzzleAttempt.id)
    )
    return build_user_records(games.all(), puzzle_attempts.all())


@router.get("/api/learning/phase-accuracy")
async def learning_phase_accuracy(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Возвращает точность ходов пользователя по этапам завершённых партий."""
    return await load_user_phase_accuracy(db, user_id=user.id)


@router.get("/api/learning/level-progress")
async def learning_level_progress(user: User = Depends(get_current_user)):
    """Положение пользователя внутри текущей рейтинговой группы."""
    return {
        **rating_band_progress(effective_player_rating(user)),
        "provisional": user.assessment_completed_at is None,
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
        player_rating=effective_player_rating(user),
    )
    return {
        "topics": public_difficulty_profile(profile, difficulties),
        "rule": {
            "min_difficulty": 1,
            "max_difficulty": 3,
            "default_difficulty": 1,
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

    valid_puzzles = [
        p for p in state.puzzle_base.get("puzzles", []) if _is_puzzle_valid(p)
    ]
    # Пробный доступ до анкеты: не больше 3 лёгких задач без персонализации.
    if _is_preview_user(user):
        today, attempted = await _preview_today_attempted(db, user.id)
        remaining = max(0, PREVIEW_PUZZLE_LIMIT - len(attempted))
        payload = _preview_payload(
            valid_puzzles,
            count,
            remaining,
            user_id=user.id,
            today=today,
            attempted=attempted,
        )
        payload["mode"] = "adaptive"
        payload["weak_topics"] = []
        payload["strong_topics"] = []
        payload["allocation"] = None
        return payload

    profile, topic_difficulties = await get_user_theme_difficulties(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
        player_rating=effective_player_rating(user),
    )
    valid_puzzles = [
        p for p in state.puzzle_base.get("puzzles", []) if _is_puzzle_valid(p)
    ]
    player_rating = effective_player_rating(user)
    max_puzzle_rating = max_puzzle_rating_for_user(player_rating)
    if player_rating is not None:
        own_group = int(rating_group_for(player_rating)["id"])
        own_pool = [p for p in valid_puzzles if puzzle_rating_group(p) == own_group]
        capped_pool = [p for p in own_pool if _puzzle_rating(p) <= max_puzzle_rating]
        # Keep the current rating group available if the ceiling would empty it.
        if capped_pool:
            valid_puzzles = capped_pool
        elif own_pool:
            valid_puzzles = own_pool
    phase_profile = await load_user_phase_accuracy(db, user_id=user.id)
    picked, allocation = select_adaptive_puzzles(
        valid_puzzles,
        weak_topics=profile["weak_topics"],
        strong_topics=profile["strong_topics"],
        count=count,
        topic_difficulties=topic_difficulties,
        phase_stats=phase_profile["phases"],
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
            "adaptive_phase": p.get("adaptive_phase"),
            "adaptive_difficulty": p.get("adaptive_difficulty"),
            "difficulty_match": p.get("difficulty_match"),
        }
        for p in picked
    ]

    player_rating = effective_player_rating(user)
    rating_group = rating_group_for(player_rating) if player_rating is not None else None

    return {
        "puzzles": result,
        "total": len(result),
        "mode": "adaptive",
        "player_level": {
            "title": rating_group["title"] if rating_group else "Стартовый",
            "rating_group": rating_group["key"] if rating_group else None,
            "provisional": user.assessment_completed_at is None,
        },
        "weak_topics": profile["weak_topics"],
        "strong_topics": profile["strong_topics"],
        "difficulty_profile": public_difficulty_profile(profile, topic_difficulties),
        "allocation": allocation,
        "phase_adjustment": allocation["phase_adjustment"],
        "selection_rule": profile["selection_rule"],
        "max_puzzle_rating": max_puzzle_rating,
    }


@router.post("/api/learning/puzzle/check")
def puzzle_check(req: PuzzleAnswerRequest):
    """Проверка отдельного тренировочного паззла с показом решения."""
    puzzle = _puzzle_lookup().get(req.puzzle_id)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    try:
        board = chess.Board(puzzle["fen"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Некорректная позиция в задаче") from exc
    solution = _solution_for(board, puzzle.get("moves", ""))
    if solution is None:
        raise HTTPException(status_code=500, detail="У задачи нет решения")

    user_move = req.move.strip()
    if len(user_move) not in (4, 5):
        user_move = _ordinal_to_uci(board, user_move) or ""
    correct = user_move == solution
    return {
        "correct": bool(correct),
        "solution": solution,
        "puzzle_rating": puzzle.get("rating", 0),
        "themes": puzzle.get("themes", []),
    }


def _base_difficulty_for_user(user: User | None) -> int:
    """Стартовая B2-сложность тем из рейтинга пользователя (тест > анкета)."""
    if user is None:
        return difficulty_for_rating(None)
    rating, _ = rating_for_selection(user.elo, user.rating_estimate, user.prior_band)
    return difficulty_for_rating(rating)


def _puzzle_rating(puzzle: dict) -> int:
    try:
        return int(puzzle.get("rating") or 0)
    except (TypeError, ValueError):
        return 0


def _sample_puzzles_by_buckets(puzzles: list[dict], count: int) -> list[dict]:
    """Режим без рейтинга: равномерно по всем рейтинговым корзинам."""
    picked: list[dict] = []
    selected_ids: set[str] = set()
    per_bucket = max(1, count // len(LEVEL_BUCKETS)) if count > 0 else 0

    for bucket in LEVEL_BUCKETS:
        lo, hi = bucket["min_puzzle_rating"], bucket["max_puzzle_rating"]
        pool = [p for p in puzzles if lo <= p.get("rating", 0) < hi]
        random.shuffle(pool)
        for puzzle in pool[:per_bucket]:
            picked.append(puzzle)
            selected_ids.add(str(puzzle.get("id", "")))

    if len(picked) < count:
        remaining = [p for p in puzzles if str(p.get("id", "")) not in selected_ids]
        random.shuffle(remaining)
        picked.extend(remaining[: max(0, count - len(picked))])
    return picked


def _shuffled_sample(puzzles: list[dict], count: int) -> list[dict]:
    """Случайная выборка до ``count`` задач из переданного пула."""
    pool = list(puzzles)
    random.shuffle(pool)
    return pool[: max(0, count)]


def _public_puzzle(puzzle: dict) -> dict:
    return {
        "id": puzzle.get("id", ""),
        "fen": puzzle.get("fen", ""),
        "moves": puzzle.get("moves", ""),
        "themes": puzzle.get("themes", []),
        "rating": puzzle.get("rating", 0),
    }


def _onboarding_completed(user: User) -> bool:
    onboarding = user.onboarding
    return isinstance(onboarding, dict) and bool(onboarding.get("submitted_at"))


def _is_preview_user(user: User | None) -> bool:
    """Зарегистрирован, но ещё не прошёл анкету (и не админ)."""
    if user is None or user.effective_role == ROLE_ADMIN:
        return False
    return not _onboarding_completed(user)


async def _preview_today_attempted(
    db: AsyncSession, user_id: int
) -> tuple[Any, set[str]]:
    """Дата на стороне БД и множество задач, решённых пользователем СЕГОДНЯ.

    День считаем так же, как и лимит — ``date(created_at) = current_date()``,
    чтобы сброс совпадал с часовым поясом, в котором пишется ``created_at``.
    """
    today = await db.scalar(select(func.current_date()))
    rows = (
        await db.scalars(
            select(distinct(PuzzleAttempt.puzzle_id)).where(
                PuzzleAttempt.user_id == user_id,
                func.date(PuzzleAttempt.created_at) == func.current_date(),
            )
        )
    ).all()
    attempted = {str(pid) for pid in rows if pid is not None}
    return today, attempted


def _preview_daily_assignment(
    valid_puzzles: list[dict],
    user_id: int,
    today: Any,
    topic: str | None,
) -> list[dict]:
    """Фиксированный на день набор пробных задач (не больше лимита).

    Порядок детерминирован (seed из id пользователя, даты и темы), поэтому при
    повторном открытии вкладки «Шахматные задачи» выдаются те же задания:
    последняя (третья) пробная задача остаётся закреплённой и не пересобирается.
    """
    pool = [p for p in valid_puzzles if _puzzle_rating(p) <= PREVIEW_MAX_RATING]
    rng = random.Random(f"{user_id}:{today}:{topic or ''}")
    rng.shuffle(pool)
    return pool[:PREVIEW_PUZZLE_LIMIT]


def _preview_payload(
    valid_puzzles: list[dict],
    count: int,
    remaining: int,
    *,
    user_id: int,
    today: Any,
    attempted: set[str],
    topic: str | None = None,
) -> dict:
    """Пробный набор: не больше ``remaining`` задач в день с рейтингом до 700.

    Набор на день фиксирован, а уже решённые сегодня задачи из него исключаются,
    поэтому нерешённые (в том числе последняя) остаются неизменными между
    заходами, пока сохраняется их дневной запас.
    """
    assigned = _preview_daily_assignment(valid_puzzles, user_id, today, topic)
    available = [p for p in assigned if str(p.get("id")) not in attempted]
    picked = available[: max(0, min(count, remaining))]
    return {
        "puzzles": [_public_puzzle(p) for p in picked],
        "total": len(picked),
        "topic": topic,
        "personalization": {
            "source": None,
            "group": None,
            "max_rating": PREVIEW_MAX_RATING,
        },
        "preview": True,
        "preview_limit": PREVIEW_PUZZLE_LIMIT,
        "preview_remaining": remaining,
        "preview_locked": remaining <= 0,
    }


@router.get("/api/learning/puzzles")
async def get_puzzles(
    count: int = Query(default=20, ge=1, le=100),
    topic: str | None = None,
    rating_min: int | None = Query(default=None, ge=0, le=10000),
    user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Возвращает набор задач с полными решениями для тренировочной страницы.

    ``topic`` — slug модуля учебного курса. Если он передан, выдаются только
    связанные с этим модулем Lichess-темы. Так TrainingPage может отправить
    пользователя сразу на релевантную практику. Практика по теме НЕ привязана
    к рейтингу: показываются все пазлы темы (иначе для редких тем и высоких
    корзин задач почти не остаётся).

    Пока пользователь не прошёл анкету, ему доступен только пробный набор:
    не больше трёх задач в день с рейтингом до 700 (см. ``_preview_payload``).
    Гость и прошедший анкету получают обычную выборку: гость — равномерно по
    корзинам, авторизованный — ТОЛЬКО свою текущую корзину с потолком
    «рейтинг + 200».

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
    if rating_min is not None:
        valid_puzzles = [
            p for p in valid_puzzles if int(p.get("rating") or 0) >= rating_min
        ]

    # Пробный доступ до анкеты: не больше 3 лёгких задач в день.
    if _is_preview_user(user):
        today, attempted = await _preview_today_attempted(db, user.id)
        remaining = max(0, PREVIEW_PUZZLE_LIMIT - len(attempted))
        return _preview_payload(
            valid_puzzles,
            count,
            remaining,
            user_id=user.id,
            today=today,
            attempted=attempted,
            topic=normalized_topic,
        )

    rating, rating_source = rating_for_selection(
        user.elo if user else None,
        user.rating_estimate if user else None,
        user.prior_band if user else None,
    )
    personalization: dict[str, Any] = {
        "source": rating_source if normalized_topic is None else None,
        "group": None,
        "max_rating": None,
    }

    if rating is not None and normalized_topic is None:
        # Только текущая корзина пользователя. Потолок «рейтинг + 200» —
        # жёсткий максимум: более сложные задачи в практике не предлагаются.
        ceiling = max_puzzle_rating_for_user(rating)
        own_group = int(rating_group_for(rating)["id"])
        own_pool = [p for p in valid_puzzles if puzzle_rating_group(p) == own_group]
        pool = [p for p in own_pool if _puzzle_rating(p) <= ceiling]
        # Потолок не должен обнулять корзину: если в пределах +200 задач нет
        # (например, у самого низа диапазона), остаёмся в своей корзине.
        if not pool:
            pool = own_pool
        picked = _shuffled_sample(pool, count)
        personalization = {
            "source": rating_source,
            "group": own_group,
            "max_rating": ceiling,
        }
    else:
        picked = _sample_puzzles_by_buckets(valid_puzzles, count)

    random.shuffle(picked)

    result = [_public_puzzle(p) for p in picked[:count]]

    return {
        "puzzles": result,
        "total": len(result),
        "topic": normalized_topic,
        "personalization": personalization,
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
