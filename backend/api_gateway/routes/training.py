"""REST API отдельной вкладки «Обучение».

Это не тактические паззлы из learning.py, а последовательный учебный курс
по правилам движения фигур и базовым шахматным понятиям.
"""

import random
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway import state
from backend.api_gateway.dependecies import get_current_user
from backend.db.session import get_db
from backend.models.training_task import TrainingTask
from backend.models.training_attempt import TrainingAttempt
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.user import User
from backend.services.gigacoin_service import (
    TRAINING_FIRST_CORRECT,
    TRAINING_REPEAT,
    active_day_streak,
    award_gigacoins,
    moscow_today,
)
from backend.services.training_checker import TrainingCheckError, check_training_task
from backend.services.assessment import rating_for_selection
from backend.services.difficulty_logic import difficulty_for_rating
from backend.services.training_reviews import (
    get_due_training_reviews,
    update_training_review_after_attempt,
)
from backend.services.dynamic_difficulty import (
    effective_player_rating,
    get_user_theme_difficulties,
    training_task_topic_slug,
    update_theme_difficulty_after_attempt,
)
from backend.services.training_service import (
    get_lesson,
    get_training_progress,
    get_module_by_slug,
    get_task,
    list_lesson_tasks,
    list_module_lessons,
    list_modules,
    save_attempt,
)

router = APIRouter(prefix="/api/training", tags=["training"])


def _base_difficulty_for_user(user: User) -> int:
    """Стартовая сложность тем из рейтинга пользователя (тест > анкета)."""
    rating, _ = rating_for_selection(user.elo, user.rating_estimate, user.prior_band)
    return difficulty_for_rating(rating)


class TrainingAnswerRequest(BaseModel):
    answer: dict[str, Any]
    hints_used: int = Field(default=0, ge=0, le=100)
    response_time_ms: int | None = Field(default=None, ge=0)


def _task_public(
    task: TrainingTask,
    correct_option_position: int | None = None,
) -> dict[str, Any]:
    """Публичная часть задания: эталон ответа намеренно не выдаётся."""
    payload = dict(task.payload or {})
    payload.pop("accepted_moves", None)
    correct_option = payload.pop("correct_option", None)

    options = payload.get("options")
    if task.task_type == "choose_option" and isinstance(options, list) and options:
        public_options = [dict(option) for option in options if isinstance(option, dict)]
        correct_option_data = next(
            (option for option in public_options if option.get("id") == correct_option),
            None,
        )
        if correct_option_data and len(public_options) > 1:
            distractors = [
                option for option in public_options
                if option.get("id") != correct_option
            ]
            random.shuffle(distractors)
            if correct_option_position is None:
                correct_option_position = random.randrange(len(public_options))
            correct_option_position = max(
                0,
                min(correct_option_position, len(public_options) - 1),
            )
            shuffled_options = []
            for index in range(len(public_options)):
                if index == correct_option_position:
                    shuffled_options.append(correct_option_data)
                else:
                    shuffled_options.append(distractors.pop())
            payload["options"] = shuffled_options

    return {
        "id": task.id,
        "lesson_id": task.lesson_id,
        "task_type": task.task_type,
        "title": task.title,
        "instruction": task.instruction,
        "fen": task.fen,
        "source_square": task.source_square,
        "difficulty": task.difficulty,
        "payload": payload,
        "sort_order": task.sort_order,
    }


def _tasks_public(tasks: list[TrainingTask]) -> list[dict[str, Any]]:
    """Смешивает позиции правильных вариантов в пределах списка заданий."""
    positions_by_count: dict[int, list[int]] = {}
    result = []

    for task in tasks:
        payload = task.payload or {}
        options = payload.get("options") or []
        correct_position = None
        if task.task_type == "choose_option" and len(options) > 1:
            option_count = len(options)
            position_pool = positions_by_count.get(option_count)
            if not position_pool:
                position_pool = list(range(option_count))
                random.shuffle(position_pool)
                positions_by_count[option_count] = position_pool
            correct_position = position_pool.pop()

        result.append(_task_public(task, correct_position))

    return result


@router.get("/progress")
async def training_progress(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Прогресс текущего пользователя по учебному курсу."""
    progress = await get_training_progress(db, user.id)
    progress["streak"] = await active_day_streak(db, user.id)
    return progress


@router.get("/due")
async def training_due(
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Задания, которые текущему пользователю уже пора повторить."""
    items = await get_due_training_reviews(db, user_id=user.id, limit=limit)
    return {
        "total": len(items),
        "items": [
            {
                "review": item["review"],
                "module": item["module"],
                "lesson": item["lesson"],
                "task": _task_public(item["task"]),
            }
            for item in items
        ],
    }


@router.get("/modules")
async def training_modules(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Все карточки учебных модулей, включая будущие disabled-модули.

    Учебный модуль доступен только зарегистрированным пользователям.
    """
    return {"modules": await list_modules(db)}


@router.get("/modules/{slug}")
async def training_module(
    slug: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    module = await get_module_by_slug(db, slug)
    if not module:
        raise HTTPException(status_code=404, detail="Учебный модуль не найден")

    return {
        "module": {
            "id": module.id,
            "slug": module.slug,
            "title": module.title,
            "description": module.description,
            "enabled": module.enabled,
            "sort_order": module.sort_order,
        },
        "lessons": await list_module_lessons(db, module.id) if module.enabled else [],
    }


@router.get("/lessons/{lesson_id}")
async def training_lesson(
    lesson_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    lesson = await get_lesson(db, lesson_id)
    if not lesson or not lesson.enabled:
        raise HTTPException(status_code=404, detail="Учебный урок не найден")

    tasks = await list_lesson_tasks(db, lesson.id)
    return {
        "lesson": {
            "id": lesson.id,
            "module_id": lesson.module_id,
            "slug": lesson.slug,
            "title": lesson.title,
            "theory": lesson.theory,
            "sort_order": lesson.sort_order,
        },
        "tasks": _tasks_public(tasks),
    }


@router.get("/tasks/{task_id}")
async def training_task(
    task_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    task = await get_task(db, task_id)
    if not task or not task.enabled:
        raise HTTPException(status_code=404, detail="Учебное задание не найдено")
    return {"task": _task_public(task)}


@router.post("/tasks/{task_id}/check")
async def check_task(
    task_id: int,
    req: TrainingAnswerRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    task = await get_task(db, task_id)
    if not task or not task.enabled:
        raise HTTPException(status_code=404, detail="Учебное задание не найдено")

    try:
        result = check_training_task(task, req.answer)
    except TrainingCheckError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    topic_slug = await training_task_topic_slug(db, task.id)
    # Инициализация до новой попытки: текущая попытка затем изменит
    # difficulty максимум на один шаг по свежей общей accuracy темы.
    await get_user_theme_difficulties(
        db,
        user_id=user.id,
        puzzle_base=state.puzzle_base,
        player_rating=effective_player_rating(user),
    )

    previously_solved = False
    if result.correct:
        previously_solved = await db.scalar(
            select(TrainingAttempt.id).where(
                TrainingAttempt.user_id == user.id,
                TrainingAttempt.task_id == task.id,
                TrainingAttempt.correct.is_(True),
            ).limit(1)
        ) is not None

    attempt = await save_attempt(
        db,
        task=task,
        user=user,
        answer=req.answer,
        result=result,
        hints_used=req.hints_used,
        response_time_ms=req.response_time_ms,
    )

    difficulty_update = await update_theme_difficulty_after_attempt(
        db,
        user_id=user.id,
        theme_slug=topic_slug,
        puzzle_base=state.puzzle_base,
        player_rating=effective_player_rating(user),
    )

    review_update = await update_training_review_after_attempt(
        db,
        user_id=user.id,
        task_id=task.id,
        correct=result.correct,
    )

    gigacoins_earned = 0
    if result.correct:
        if previously_solved:
            today = moscow_today().isoformat()
            reward_reason = "training_repeat"
            reward_ref = f"{task.id}:{today}"
            reward_amount = TRAINING_REPEAT
        else:
            reward_reason = "training_first"
            reward_ref = str(task.id)
            reward_amount = TRAINING_FIRST_CORRECT
        gigacoins_earned += await award_gigacoins(
            db, user, reward_reason, reward_ref, reward_amount
        )

        module_id = await db.scalar(
            select(TrainingLesson.module_id)
            .join(TrainingTask, TrainingTask.lesson_id == TrainingLesson.id)
            .where(TrainingTask.id == task.id)
        )
        module = await db.get(TrainingModule, module_id) if module_id else None
        if module is not None and module.enabled:
            module_task_ids = set((await db.scalars(
                select(TrainingTask.id)
                .join(TrainingLesson, TrainingLesson.id == TrainingTask.lesson_id)
                .where(
                    TrainingLesson.module_id == module.id,
                    TrainingLesson.enabled.is_(True),
                    TrainingTask.enabled.is_(True),
                )
            )).all())
            solved_task_ids = set((await db.scalars(
                select(TrainingAttempt.task_id)
                .where(
                    TrainingAttempt.user_id == user.id,
                    TrainingAttempt.correct.is_(True),
                    TrainingAttempt.task_id.in_(module_task_ids or {-1}),
                )
                .distinct()
            )).all())
            if module_task_ids and module_task_ids.issubset(solved_task_ids):
                gigacoins_earned += await award_gigacoins(
                    db, user, "module_complete", str(module.id), 10
                )
        await db.commit()
    else:
        gigacoins_earned += await award_gigacoins(
            db, user, "training_activity", str(attempt.id), 0
        )
        await db.commit()

    return {
        "ok": True,
        "task_id": task.id,
        "attempt_id": attempt.id,
        "attempt_number": attempt.attempt_number,
        "hints_used": attempt.hints_used,
        "correct": result.correct,
        "topic": topic_slug,
        "difficulty_update": difficulty_update,
        "review_update": review_update,
        "score": result.score,
        "feedback": result.feedback,
        "explanation": task.explanation,
        "expected": result.expected,
        "details": result.details,
        "gigacoins_earned": gigacoins_earned,
        "gigacoins_balance": int(user.gigacoins or 0),
    }
