"""B2: динамическая сложность пользователя по темам курса.

Правило ТЗ:
- accuracy > 80% -> difficulty + 1
- accuracy < 50% -> difficulty - 1
- 50% <= accuracy <= 80% -> без изменения

Шкала 1..3 совпадает с difficulty учебных заданий TrainingTask.
Нейтральный стартовый уровень — 2, чтобы правило могло работать в обе стороны.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask
from backend.models.user_theme_difficulty import UserThemeDifficulty
from backend.services.adaptive_training import build_weakness_profile
from backend.services.course_topics import COURSE_TOPIC_META
from backend.services.difficulty_logic import (
    DEFAULT_DIFFICULTY,
    next_difficulty,
    rating_band_for_difficulty,
)


def _accuracy_by_slug(profile: dict) -> dict[str, float | None]:
    return {
        str(topic.get("slug")): topic.get("accuracy")
        for topic in profile.get("topics", [])
        if topic.get("slug")
    }


async def ensure_user_theme_difficulties(
    db: AsyncSession,
    *,
    user_id: int,
    profile: dict,
) -> dict[str, int]:
    """Создаёт отсутствующие строки difficulty один раз.

    Для уже накопленной до B2 истории стартуем с нейтрального уровня 2 и
    однократно применяем текущее accuracy. Повторные GET не меняют difficulty.
    PostgreSQL ON CONFLICT защищает ленивую инициализацию от гонки.
    """
    existing = (
        await db.scalars(
            select(UserThemeDifficulty).where(UserThemeDifficulty.user_id == user_id)
        )
    ).all()
    existing_slugs = {row.theme_slug for row in existing}
    accuracy_by_slug = _accuracy_by_slug(profile)

    inserted = False
    for slug in COURSE_TOPIC_META:
        if slug in existing_slugs:
            continue
        initial = next_difficulty(DEFAULT_DIFFICULTY, accuracy_by_slug.get(slug))
        stmt = (
            pg_insert(UserThemeDifficulty)
            .values(
                user_id=user_id,
                theme_slug=slug,
                current_difficulty=initial,
            )
            .on_conflict_do_nothing(constraint="uq_user_theme_difficulty")
        )
        await db.execute(stmt)
        inserted = True

    if inserted:
        await db.commit()

    rows = (
        await db.scalars(
            select(UserThemeDifficulty).where(UserThemeDifficulty.user_id == user_id)
        )
    ).all()
    return {row.theme_slug: int(row.current_difficulty) for row in rows}


async def get_user_theme_difficulties(
    db: AsyncSession,
    *,
    user_id: int,
    puzzle_base: dict | None,
) -> tuple[dict, dict[str, int]]:
    profile = await build_weakness_profile(
        db,
        user_id=user_id,
        puzzle_base=puzzle_base,
    )
    difficulties = await ensure_user_theme_difficulties(
        db,
        user_id=user_id,
        profile=profile,
    )
    return profile, difficulties


async def training_task_topic_slug(db: AsyncSession, task_id: int) -> str | None:
    return await db.scalar(
        select(TrainingModule.slug)
        .select_from(TrainingTask)
        .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
        .join(TrainingModule, TrainingLesson.module_id == TrainingModule.id)
        .where(TrainingTask.id == task_id)
    )


async def update_theme_difficulty_after_attempt(
    db: AsyncSession,
    *,
    user_id: int,
    theme_slug: str | None,
    puzzle_base: dict | None,
) -> dict | None:
    """После новой попытки двигает difficulty темы максимум на один шаг.

    Строка блокируется SELECT FOR UPDATE, поэтому параллельные попытки одного
    пользователя по одной теме не теряют инкременты/декременты.
    """
    if not theme_slug or theme_slug not in COURSE_TOPIC_META:
        return None

    # Строка обычно уже создана ensure_user_theme_difficulties(). Но оставляем
    # безопасную ленивую вставку для прямых вызовов сервиса.
    stmt = (
        pg_insert(UserThemeDifficulty)
        .values(
            user_id=user_id,
            theme_slug=theme_slug,
            current_difficulty=DEFAULT_DIFFICULTY,
        )
        .on_conflict_do_nothing(constraint="uq_user_theme_difficulty")
    )
    await db.execute(stmt)
    await db.flush()

    row = await db.scalar(
        select(UserThemeDifficulty)
        .where(
            UserThemeDifficulty.user_id == user_id,
            UserThemeDifficulty.theme_slug == theme_slug,
        )
        .with_for_update()
    )
    if row is None:
        return None

    profile = await build_weakness_profile(
        db,
        user_id=user_id,
        puzzle_base=puzzle_base,
    )
    topic = next(
        (item for item in profile.get("topics", []) if item.get("slug") == theme_slug),
        None,
    )
    accuracy = topic.get("accuracy") if topic else None

    previous = int(row.current_difficulty)
    current = next_difficulty(previous, accuracy)
    row.current_difficulty = current
    row.updated_at = datetime.now(UTC)
    await db.commit()

    return {
        "theme_slug": theme_slug,
        "accuracy": accuracy,
        "previous_difficulty": previous,
        "current_difficulty": current,
        "changed": current != previous,
    }


def public_difficulty_profile(profile: dict, difficulties: dict[str, int]) -> list[dict]:
    result: list[dict] = []
    for topic in profile.get("topics", []):
        slug = topic.get("slug")
        difficulty = int(difficulties.get(slug, DEFAULT_DIFFICULTY))
        min_rating, max_rating = rating_band_for_difficulty(difficulty)
        result.append(
            {
                "slug": slug,
                "title": topic.get("title"),
                "accuracy": topic.get("accuracy"),
                "attempts": topic.get("attempts", 0),
                "current_difficulty": difficulty,
                "puzzle_rating_band": {
                    "min": min_rating,
                    "max": max_rating,
                },
            }
        )
    return result
