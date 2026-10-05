"""Обучение, стартовая оценка и адаптивные тактические задания."""

from __future__ import annotations

import random
import secrets
from datetime import UTC, datetime
from typing import Any, Literal

import chess
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway import state
from backend.api_gateway.dependecies import get_current_user
from backend.api_gateway.models import PuzzleAnswerRequest
from backend.db.session import get_db
from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.user import ROLE_ADMIN, User
from backend.models.user_chess_account import UserChessAccount
from backend.services.adaptive_logic import select_adaptive_puzzles
from backend.services.adaptive_training import (
    build_weakness_profile,
    compute_learning_progress,
)
from backend.services.assessment import (
    RATING_GROUPS,
    build_onboarding_feedback,
    build_test_feedback,
    calculate_performance_rating,
    onboarding_rating,
    personalized_level_test_puzzles,
    puzzle_rating_group,
    rating_group_for,
)
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


# Общие рейтинговые корзины для обычной страницы паззлов. Границы совпадают
# с новой стартовой оценкой: 0–1000 / 1001–1500 / 1501–1900 / 1901+.
LEVEL_BUCKETS = [
    {"min_puzzle_rating": 0, "max_puzzle_rating": 1001},
    {"min_puzzle_rating": 1001, "max_puzzle_rating": 1501},
    {"min_puzzle_rating": 1501, "max_puzzle_rating": 1901},
    {"min_puzzle_rating": 1901, "max_puzzle_rating": 10000},
]


class OnboardingQ2(BaseModel):
    has_rating: bool = False
    platform: Literal["lichess", "chesscom"] | None = None
    username: str | None = Field(default=None, max_length=64)
    # Эти поля принимает старый клиент, но новый backend не доверяет им:
    # при отправке анкеты рейтинг заново берётся из user_chess_accounts.
    rating_type: Literal["blitz", "rapid", "bullet"] | None = None
    rating: int | None = Field(default=None, ge=0, le=3500)
    rating_scale: str | None = Field(default=None, max_length=64)
    rating_usable: bool | None = None
    linked_account: bool | None = None

    @model_validator(mode="after")
    def validate_external_account(self):
        if self.has_rating and (not self.platform or not self.username):
            raise ValueError("Для внешнего рейтинга укажите платформу и имя связанного аккаунта")
        return self


class OnboardingRequest(BaseModel):
    q1: Literal["never", "know_moves", "sometimes", "regularly", "tournaments"]
    q2: OnboardingQ2 = Field(default_factory=OnboardingQ2)
    q3: list[Literal["friends_family", "online_rating", "tournaments", "child"]] = Field(
        min_length=1, max_length=2
    )
    q4: Literal["lt1", "1_3", "3_5", "5plus"]
    q5: list[Literal["puzzles", "games", "lessons"]] = Field(min_length=3, max_length=3)
    q6: Literal["yes", "stuck", "no"]
    q7: Literal["under10", "10_16", "17plus"]
    q8: Literal["coach", "friends", "internet", "school", "other"] | None = None
    parental_consent: bool = False
    guardian_contact: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_questionnaire(self):
        if len(set(self.q3)) != len(self.q3):
            raise ValueError("Q3 не должен содержать повторяющиеся цели")
        if len(set(self.q5)) != 3:
            raise ValueError("Q5 должен содержать уникальный порядок из трёх форматов")
        if self.q7 == "under10" and not self.parental_consent:
            raise ValueError("Для группы до 10 лет требуется согласие родителя/опекуна")
        return self


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


def _upsert_test_answer(test: LevelTest, answer: dict[str, Any]) -> None:
    answers = [
        item
        for item in (test.answers or [])
        if str(item.get("puzzle_id")) != str(answer.get("puzzle_id"))
    ]
    answers.append(answer)
    test.answers = answers


@router.get("/api/learning/assessment/status")
async def assessment_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Состояние обязательной стартовой оценки для глобального уведомления."""
    if user.effective_role == ROLE_ADMIN:
        return {
            "required": False,
            "phase": "not_applicable",
            "reason": "admin",
            "onboarding_completed": False,
            "level_test_completed": False,
        }

    onboarding_completed = bool(
        isinstance(user.onboarding, dict) and user.onboarding.get("submitted_at")
    )
    completed = user.assessment_completed_at is not None
    active = None if completed else await _latest_personal_test(db, user.id)

    if completed:
        phase = "completed"
    elif not onboarding_completed:
        phase = "onboarding"
    else:
        phase = "level_test"

    return {
        "required": not completed,
        "phase": phase,
        "onboarding_completed": onboarding_completed,
        "level_test_completed": completed,
        "rating_estimate": user.rating_estimate,
        "rating_group": rating_group_for(user.rating_estimate)["key"] if user.rating_estimate is not None else None,
        "skill_band": user.skill_band,
        "elo": user.elo,
        "active_test": (
            {
                "test_id": active.id,
                "answered": len(active.answers or []),
                "total": len(active.question_ids or []),
            }
            if active
            else None
        ),
    }


@router.post("/api/learning/assessment/onboarding")
async def assessment_onboarding(
    req: OnboardingRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user.effective_role == ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Администратору стартовая оценка не требуется")
    if user.assessment_completed_at is not None:
        raise HTTPException(status_code=409, detail="Стартовая оценка уже завершена")

    answers = req.model_dump()

    # Внешнему рейтингу из тела запроса не доверяем. Если Q2 включён, берём
    # рейтинг только из сохранённой soft-link записи, созданной через
    # POST /api/chess-profile/link. Это не даёт вручную подменить Elo в DevTools.
    if req.q2.has_rating:
        linked = await db.scalar(
            select(UserChessAccount).where(
                UserChessAccount.user_id == user.id,
                UserChessAccount.platform == req.q2.platform,
            )
        )
        if linked is None:
            raise HTTPException(
                status_code=409,
                detail="Сначала найдите и привяжите шахматный аккаунт в вопросе Q2",
            )
        if (linked.username or "").casefold() != (req.q2.username or "").strip().casefold():
            raise HTTPException(
                status_code=409,
                detail="Указанный профиль не совпадает с сохранённой привязкой. Привяжите аккаунт заново.",
            )
        answers["q2"] = {
            "has_rating": True,
            "platform": linked.platform,
            "username": linked.username,
            "rating_type": linked.rating_type,
            "rating": linked.rating if linked.rating_usable else None,
            "rating_scale": linked.rating_scale,
            "rating_usable": bool(linked.rating_usable),
            "linked_account": True,
            "verified": bool(linked.verified),
            "games": linked.games,
            "rating_deviation": linked.rating_deviation,
        }
    else:
        answers["q2"] = {
            "has_rating": False,
            "platform": None,
            "username": None,
            "rating_type": None,
            "rating": None,
            "rating_scale": None,
            "rating_usable": False,
            "linked_account": False,
            "verified": False,
        }

    rating, scale, prior_band = onboarding_rating(answers)
    feedback = build_onboarding_feedback(answers, rating)

    locked_user = await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if locked_user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    locked_user.rating_estimate = rating
    locked_user.rating_scale = scale
    locked_user.prior_band = prior_band
    locked_user.onboarding = {
        "answers": answers,
        "feedback": feedback,
        "submitted_at": datetime.now(UTC).isoformat(),
        "version": 3,
    }
    await db.commit()

    group = rating_group_for(rating)
    return {
        "ok": True,
        "rating_estimate": rating,
        "rating_scale": scale,
        "prior_band": prior_band,
        "rating_group": group,
        "feedback": feedback,
    }


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

    rating_estimate = int(user.rating_estimate or user.elo or 800)
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
    return await compute_learning_progress(db, user_id=user.id)


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
