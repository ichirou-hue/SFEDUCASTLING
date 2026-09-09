"""REST API отдельной вкладки «Обучение».

Это не тактические паззлы из learning.py, а последовательный учебный курс
по правилам движения фигур и базовым шахматным понятиям.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api_gateway.dependecies import get_optional_current_user
from backend.db.session import get_db
from backend.models.training_task import TrainingTask
from backend.models.user import User
from backend.services.training_checker import TrainingCheckError, check_training_task
from backend.services.training_service import (
    get_lesson,
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

    attempt = await save_attempt(
        db,
        task=task,
        user=user,
        answer=req.answer,
        result=result,
        hints_used=req.hints_used,
        response_time_ms=req.response_time_ms,
    )

    return {
        "ok": True,
        "task_id": task.id,
        "attempt_id": attempt.id,
        "attempt_number": attempt.attempt_number,
        "correct": result.correct,
        "score": result.score,
        "feedback": result.feedback,
        "explanation": task.explanation,
        "expected": result.expected,
        "details": result.details,
    }
