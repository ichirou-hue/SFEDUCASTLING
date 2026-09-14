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


def _current_streak(activity_dates: set, today=None) -> int:
    """Возвращает текущую серию дней с учебной активностью.

    Серия не обнуляется утром нового дня: если сегодня попыток ещё не было,
    но активность была вчера, отсчёт продолжается от вчерашней даты.
    """
    from datetime import date, timedelta

    if not activity_dates:
        return 0

    today = today or date.today()
    if today in activity_dates:
        cursor = today
    elif today - timedelta(days=1) in activity_dates:
        cursor = today - timedelta(days=1)
    else:
        return 0

    streak = 0
    while cursor in activity_dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


async def get_training_progress(db: AsyncSession, user_id: int) -> dict:
    """Агрегирует прогресс пользователя по учебному курсу.

    Модуль считается пройденным, когда пользователь хотя бы один раз
    правильно выполнил каждое включённое задание этого модуля.
    Accuracy считается по всем строкам training_attempts, включая повторы.
    """
    modules = (
        await db.scalars(
            select(TrainingModule)
            .where(TrainingModule.enabled.is_(True))
            .order_by(TrainingModule.sort_order, TrainingModule.id)
        )
    ).all()

    module_ids = [module.id for module in modules]
    tasks_by_module: dict[int, set[int]] = {module.id: set() for module in modules}

    if module_ids:
        task_rows = (
            await db.execute(
                select(TrainingTask.id, TrainingLesson.module_id)
                .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
                .where(
                    TrainingLesson.module_id.in_(module_ids),
                    TrainingLesson.enabled.is_(True),
                    TrainingTask.enabled.is_(True),
                )
            )
        ).all()
        for task_id, module_id in task_rows:
            tasks_by_module[module_id].add(task_id)

    attempt_rows = (
        await db.execute(
            select(
                TrainingAttempt.task_id,
                TrainingAttempt.correct,
                TrainingAttempt.created_at,
                TrainingLesson.module_id,
            )
            .join(TrainingTask, TrainingAttempt.task_id == TrainingTask.id)
            .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
            .join(TrainingModule, TrainingLesson.module_id == TrainingModule.id)
            .where(
                TrainingAttempt.user_id == user_id,
                TrainingModule.enabled.is_(True),
                TrainingLesson.enabled.is_(True),
                TrainingTask.enabled.is_(True),
            )
        )
    ).all()

    correct_task_ids: set[int] = set()
    topic_stats: dict[int, dict[str, int]] = {
        module.id: {"attempts": 0, "correct": 0} for module in modules
    }
    activity_dates = set()

    total_attempts = 0
    total_correct = 0
    for task_id, correct, created_at, module_id in attempt_rows:
        total_attempts += 1
        if correct:
            total_correct += 1
            correct_task_ids.add(task_id)

        if module_id in topic_stats:
            topic_stats[module_id]["attempts"] += 1
            if correct:
                topic_stats[module_id]["correct"] += 1

        if created_at is not None:
            activity_dates.add(created_at.date())

    completed_modules = 0
    topics = []
    for module in modules:
        task_ids = tasks_by_module.get(module.id, set())
        completed = bool(task_ids) and task_ids.issubset(correct_task_ids)
        if completed:
            completed_modules += 1

        stats = topic_stats[module.id]
        attempts = stats["attempts"]
        correct = stats["correct"]
        accuracy = round(correct * 100 / attempts, 1) if attempts else None
        topics.append(
            {
                "module_id": module.id,
                "slug": module.slug,
                "title": module.title,
                "attempts": attempts,
                "correct": correct,
                "accuracy": accuracy,
                "completed": completed,
            }
        )

    total_modules = len(modules)
    module_percent = round(completed_modules * 100 / total_modules, 1) if total_modules else 0.0
    overall_accuracy = round(total_correct * 100 / total_attempts, 1) if total_attempts else 0.0

    return {
        "modules": {
            "completed": completed_modules,
            "total": total_modules,
            "percent": module_percent,
        },
        "attempts": {
            "total": total_attempts,
            "correct": total_correct,
            "accuracy": overall_accuracy,
        },
        "topics": topics,
        "streak": _current_streak(activity_dates),
    }
