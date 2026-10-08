"""B3: хранение и выдача интервальных повторений учебного курса."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_review import TrainingReview
from backend.models.training_task import TrainingTask
from backend.services.review_logic import DEFAULT_EASE, next_review_state


async def update_training_review_after_attempt(
    db: AsyncSession,
    *,
    user_id: int,
    task_id: int,
    correct: bool,
    now: datetime | None = None,
) -> dict:
    """Создаёт/обновляет review после попытки и возвращает публичное состояние.

    Строка блокируется SELECT FOR UPDATE, чтобы параллельные ответы по одному
    заданию не потеряли reps/next_review_at.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)

    stmt = (
        pg_insert(TrainingReview)
        .values(
            user_id=user_id,
            task_id=task_id,
            next_review_at=current,
            ease=DEFAULT_EASE,
            reps=0,
        )
        .on_conflict_do_nothing(constraint="uq_training_review_user_task")
    )
    await db.execute(stmt)
    await db.flush()

    row = await db.scalar(
        select(TrainingReview)
        .where(
            TrainingReview.user_id == user_id,
            TrainingReview.task_id == task_id,
        )
        .with_for_update()
    )
    if row is None:  # pragma: no cover - defensive branch
        raise RuntimeError("Не удалось создать training_review")

    previous_reps = int(row.reps)
    previous_next_review_at = row.next_review_at
    state = next_review_state(
        reps=previous_reps,
        ease=float(row.ease),
        correct=bool(correct),
        now=current,
    )

    row.reps = int(state["reps"])
    row.ease = float(state["ease"])
    row.next_review_at = state["next_review_at"]
    row.updated_at = current
    await db.commit()

    return {
        "task_id": task_id,
        "correct": bool(correct),
        "previous_reps": previous_reps,
        "reps": row.reps,
        "ease": row.ease,
        "interval_days": state["interval_days"],
        "previous_next_review_at": (
            previous_next_review_at.isoformat() if previous_next_review_at else None
        ),
        "next_review_at": row.next_review_at.isoformat(),
    }


async def get_due_training_reviews(
    db: AsyncSession,
    *,
    user_id: int,
    limit: int = 50,
    now: datetime | None = None,
) -> list[dict]:
    """Возвращает просроченные и уже наступившие повторения пользователя."""
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)

    rows = (
        await db.execute(
            select(TrainingReview, TrainingTask, TrainingLesson, TrainingModule)
            .join(TrainingTask, TrainingReview.task_id == TrainingTask.id)
            .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
            .join(TrainingModule, TrainingLesson.module_id == TrainingModule.id)
            .where(
                TrainingReview.user_id == user_id,
                TrainingReview.next_review_at <= current,
                TrainingTask.enabled.is_(True),
                TrainingLesson.enabled.is_(True),
                TrainingModule.enabled.is_(True),
            )
            .order_by(TrainingReview.next_review_at.asc(), TrainingReview.id.asc())
            .limit(limit)
        )
    ).all()

    result: list[dict] = []
    for review, task, lesson, module in rows:
        overdue_seconds = max(
            0,
            int((current - review.next_review_at).total_seconds()),
        )
        result.append(
            {
                "review": {
                    "id": review.id,
                    "task_id": review.task_id,
                    "next_review_at": review.next_review_at.isoformat(),
                    "ease": float(review.ease),
                    "reps": int(review.reps),
                    "overdue_seconds": overdue_seconds,
                },
                "module": {
                    "id": module.id,
                    "slug": module.slug,
                    "title": module.title,
                },
                "lesson": {
                    "id": lesson.id,
                    "slug": lesson.slug,
                    "title": lesson.title,
                },
                "task": task,
            }
        )
    return result
