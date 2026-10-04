"""REST API отдельной вкладки «Обучение».

Это не тактические паззлы из learning.py, а последовательный учебный курс
по правилам движения фигур и базовым шахматным понятиям.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway import state
from backend.api_gateway.dependecies import get_current_user, get_optional_current_user
from backend.db.session import get_db
from backend.models.training_task import TrainingTask
from backend.models.user import User
from backend.services.training_checker import TrainingCheckError, check_training_task
from backend.services.training_reviews import (
    get_due_training_reviews,
    update_training_review_after_attempt,
)
from backend.services.dynamic_difficulty import (
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


class TrainingAnswerRequest(BaseModel):
    answer: dict[str, Any]
    hints_used: int = Field(default=0, ge=0, le=100)
    response_time_ms: int | None = Field(default=None, ge=0)


def _task_public(task: TrainingTask) -> dict[str, Any]:
    """Публичная часть задания: эталон ответа намеренно не выдаётся."""
    payload = dict(task.payload or {})
    payload.pop("accepted_moves", None)
    payload.pop("correct_option", None)

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


@router.get("/progress")
async def training_progress(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Прогресс текущего пользователя по учебному курсу."""
    return await get_training_progress(db, user.id)


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
async def training_modules(db: AsyncSession = Depends(get_db)):
    """Все карточки учебных модулей, включая будущие disabled-модули."""
    return {"modules": await list_modules(db)}


@router.get("/modules/{slug}")
async def training_module(slug: str, db: AsyncSession = Depends(get_db)):
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
async def training_lesson(lesson_id: int, db: AsyncSession = Depends(get_db)):
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
        "tasks": [_task_public(task) for task in tasks],
    }


@router.get("/tasks/{task_id}")
async def training_task(task_id: int, db: AsyncSession = Depends(get_db)):
    task = await get_task(db, task_id)
    if not task or not task.enabled:
        raise HTTPException(status_code=404, detail="Учебное задание не найдено")
    return {"task": _task_public(task)}


@router.post("/tasks/{task_id}/check")
async def check_task(
    task_id: int,
    req: TrainingAnswerRequest,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_optional_current_user),
):
    task = await get_task(db, task_id)
    if not task or not task.enabled:
        raise HTTPException(status_code=404, detail="Учебное задание не найдено")

    try:
        result = check_training_task(task, req.answer)
    except TrainingCheckError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    topic_slug = None
    if user is not None:
        topic_slug = await training_task_topic_slug(db, task.id)
        # Инициализация до новой попытки: текущая попытка затем изменит
        # difficulty максимум на один шаг по свежей общей accuracy темы.
        await get_user_theme_difficulties(
            db,
            user_id=user.id,
            puzzle_base=state.puzzle_base,
        )

    attempt = await save_attempt(
        db,
        task=task,
        user=user,
        answer=req.answer,
        result=result,
        hints_used=req.hints_used,
        response_time_ms=req.response_time_ms,
    )

    difficulty_update = None
    review_update = None
    if user is not None:
        difficulty_update = await update_theme_difficulty_after_attempt(
            db,
            user_id=user.id,
            theme_slug=topic_slug,
            puzzle_base=state.puzzle_base,
        )
        review_update = await update_training_review_after_attempt(
            db,
            user_id=user.id,
            task_id=task.id,
            correct=result.correct,
        )

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
    }
