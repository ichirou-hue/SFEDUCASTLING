"""Сервисный слой учебной подсистемы."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.training_attempt import TrainingAttempt
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask
from backend.models.user import User
from backend.services.training_checker import TrainingCheckResult


async def list_modules(db: AsyncSession) -> list[dict]:
    modules = (
        await db.scalars(select(TrainingModule).order_by(TrainingModule.sort_order, TrainingModule.id))
    ).all()

    result: list[dict] = []
    for module in modules:
        lesson_count = await db.scalar(
            select(func.count(TrainingLesson.id)).where(
                TrainingLesson.module_id == module.id,
                TrainingLesson.enabled.is_(True),
            )
        )
        task_count = await db.scalar(
            select(func.count(TrainingTask.id))
            .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
            .where(
                TrainingLesson.module_id == module.id,
                TrainingLesson.enabled.is_(True),
                TrainingTask.enabled.is_(True),
            )
        )
        result.append(
            {
                "id": module.id,
                "slug": module.slug,
                "title": module.title,
                "description": module.description,
                "sort_order": module.sort_order,
                "enabled": module.enabled,
                "lesson_count": int(lesson_count or 0),
                "task_count": int(task_count or 0),
            }
        )
    return result


async def get_module_by_slug(db: AsyncSession, slug: str) -> TrainingModule | None:
    return await db.scalar(select(TrainingModule).where(TrainingModule.slug == slug))


async def list_module_lessons(db: AsyncSession, module_id: int) -> list[dict]:
    lessons = (
        await db.scalars(
            select(TrainingLesson)
            .where(TrainingLesson.module_id == module_id, TrainingLesson.enabled.is_(True))
            .order_by(TrainingLesson.sort_order, TrainingLesson.id)
        )
    ).all()

    result: list[dict] = []
    for lesson in lessons:
        task_count = await db.scalar(
            select(func.count(TrainingTask.id)).where(
                TrainingTask.lesson_id == lesson.id,
                TrainingTask.enabled.is_(True),
            )
        )
        result.append(
            {
                "id": lesson.id,
                "slug": lesson.slug,
                "title": lesson.title,
                "sort_order": lesson.sort_order,
                "task_count": int(task_count or 0),
            }
        )
    return result


async def get_lesson(db: AsyncSession, lesson_id: int) -> TrainingLesson | None:
    return await db.get(TrainingLesson, lesson_id)


async def list_lesson_tasks(db: AsyncSession, lesson_id: int) -> list[TrainingTask]:
    return (
        await db.scalars(
            select(TrainingTask)
            .where(TrainingTask.lesson_id == lesson_id, TrainingTask.enabled.is_(True))
            .order_by(TrainingTask.sort_order, TrainingTask.id)
        )
    ).all()


async def get_task(db: AsyncSession, task_id: int) -> TrainingTask | None:
    return await db.get(TrainingTask, task_id)


async def save_attempt(
    db: AsyncSession,
    *,
    task: TrainingTask,
    user: User | None,
    answer: dict,
    result: TrainingCheckResult,
    hints_used: int,
    response_time_ms: int | None,
) -> TrainingAttempt:
    attempt_number = 1
    if user is not None:
        previous = await db.scalar(
            select(func.count(TrainingAttempt.id)).where(
                TrainingAttempt.user_id == user.id,
                TrainingAttempt.task_id == task.id,
            )
        )
        attempt_number = int(previous or 0) + 1

    attempt = TrainingAttempt(
        user_id=user.id if user else None,
        task_id=task.id,
        answer=answer,
        correct=result.correct,
        score=result.score,
        attempt_number=attempt_number,
        hints_used=hints_used,
        response_time_ms=response_time_ms,
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)
    return attempt
