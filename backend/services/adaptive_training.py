"""Агрегация учебных попыток для персонального подбора B1."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.training_attempt import TrainingAttempt
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask
from backend.services.adaptive_logic import rank_topic_profile
from backend.services.course_topics import (
    COURSE_TOPIC_META,
    primary_course_topic_for_puzzle,
)


async def build_weakness_profile(
    db: AsyncSession,
    *,
    user_id: int,
    puzzle_base: dict | None,
) -> dict:
    """Агрегирует training_attempts и puzzle_attempts по темам курса."""
    module_rows = (
        await db.execute(
            select(TrainingModule.slug, TrainingModule.title)
            .where(TrainingModule.enabled.is_(True))
            .order_by(TrainingModule.sort_order, TrainingModule.id)
        )
    ).all()

    ordered_slugs = [slug for slug, _title in module_rows if slug in COURSE_TOPIC_META]
    if not ordered_slugs:
        ordered_slugs = list(COURSE_TOPIC_META)

    module_titles = {slug: title for slug, title in module_rows}

    stats: dict[str, dict[str, int]] = {
        slug: {
            "training_attempts": 0,
            "training_correct": 0,
            "puzzle_attempts": 0,
            "puzzle_correct": 0,
        }
        for slug in ordered_slugs
    }

    training_rows = (
        await db.execute(
            select(TrainingModule.slug, TrainingAttempt.correct)
            .select_from(TrainingAttempt)
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

    for slug, correct in training_rows:
        if slug not in stats:
            continue
        stats[slug]["training_attempts"] += 1
        if correct:
            stats[slug]["training_correct"] += 1

    puzzle_lookup = {
        str(p.get("id", "")): p for p in (puzzle_base or {}).get("puzzles", [])
    }
    puzzle_rows = (
        await db.execute(
            select(PuzzleAttempt.puzzle_id, PuzzleAttempt.correct).where(
                PuzzleAttempt.user_id == user_id
            )
        )
    ).all()

    unmapped_puzzle_attempts = 0
    for puzzle_id, correct in puzzle_rows:
        slug = primary_course_topic_for_puzzle(puzzle_lookup.get(str(puzzle_id)))
        if not slug or slug not in stats:
            unmapped_puzzle_attempts += 1
            continue
        stats[slug]["puzzle_attempts"] += 1
        if correct:
            stats[slug]["puzzle_correct"] += 1

    topics: list[dict] = []
    for slug in ordered_slugs:
        s = stats[slug]
        attempts = s["training_attempts"] + s["puzzle_attempts"]
        correct = s["training_correct"] + s["puzzle_correct"]
        accuracy = round(correct * 100 / attempts, 1) if attempts else None
        topics.append(
            {
                "slug": slug,
                "title": module_titles.get(slug) or COURSE_TOPIC_META[slug]["title"],
                "attempts": attempts,
                "correct": correct,
                "accuracy": accuracy,
                "training": {
                    "attempts": s["training_attempts"],
                    "correct": s["training_correct"],
                },
                "puzzles": {
                    "attempts": s["puzzle_attempts"],
                    "correct": s["puzzle_correct"],
                },
            }
        )

    weak_topics, strong_topics = rank_topic_profile(topics)
    return {
        "topics": topics,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "unmapped_puzzle_attempts": unmapped_puzzle_attempts,
        "selection_rule": {
            "weak_share": 0.70,
            "strong_share": 0.30,
            "weak_topic_count": 3,
        },
    }
