"""Наполнение тестовых интервальных повторений для проверки UI.

Запуск из корня проекта:
    python -m scripts.seed_due_reviews strayedmoon --count 6
    python -m scripts.seed_due_reviews strayedmoon --clear

Скрипт работает только с интервальными повторениями: создаёт новые строки
``training_reviews`` для включённых заданий или сдвигает существующие в
прошлое. Уроки, модули и прогресс пользователя не меняются. Повторения по
отключённым урокам остаются нетронутыми и в выдачу /api/training/due не
попадают.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from backend.db.session import async_session_factory
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_review import TrainingReview
from backend.models.training_task import TrainingTask
from backend.models.user import User

# Как в backend/services/review_logic.py: базовая сложность SM-2.
DEFAULT_EASE = 2.2


async def _find_user(db, login: str) -> User:
    user = await db.scalar(select(User).where(User.login == login))
    if user is None:
        raise SystemExit(f"Пользователь '{login}' не найден.")
    return user


def _enabled_tasks_stmt():
    return (
        select(TrainingTask)
        .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
        .join(TrainingModule, TrainingLesson.module_id == TrainingModule.id)
        .where(
            TrainingTask.enabled.is_(True),
            TrainingLesson.enabled.is_(True),
            TrainingModule.enabled.is_(True),
        )
        .order_by(
            TrainingModule.sort_order,
            TrainingLesson.sort_order,
            TrainingTask.sort_order,
            TrainingTask.id,
        )
    )


async def seed_due_reviews(login: str, count: int, days_back: int) -> int:
    now = datetime.now(UTC).replace(tzinfo=None)

    async with async_session_factory() as db:
        user = await _find_user(db, login)
        tasks = list((await db.execute(_enabled_tasks_stmt())).scalars())
        if not tasks:
            print("Нет включённых заданий: нечего добавлять в повторения.")
            return 1

        reviews = list(
            (
                await db.execute(
                    select(TrainingReview).where(
                        TrainingReview.user_id == user.id,
                    )
                )
            ).scalars()
        )
        reviews_by_task = {review.task_id: review for review in reviews}

        # Сначала двигаем уже существующие повторения по доступным заданиям,
        # затем добираем новые. Так сохраняются reps/ease реальных повторений.
        ordered = [
            task for task in tasks if task.id in reviews_by_task
        ] + [
            task for task in tasks if task.id not in reviews_by_task
        ]
        selected = ordered[:count]

        created = 0
        updated = 0
        for index, task in enumerate(selected):
            due_at = now - timedelta(days=days_back, minutes=10 * index)
            review = reviews_by_task.get(task.id)
            if review is None:
                db.add(
                    TrainingReview(
                        user_id=user.id,
                        task_id=task.id,
                        next_review_at=due_at,
                        ease=DEFAULT_EASE,
                        reps=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
                created += 1
            else:
                review.next_review_at = due_at
                review.updated_at = now
                updated += 1

        await db.commit()

        print(
            f"Пользователь '{login}' (id={user.id}): "
            f"создано {created}, обновлено {updated} повторений."
        )
        for task in selected:
            print(f"  - task_id={task.id}: {task.title}")

        if len(selected) < count:
            print(
                f"Доступных включённых заданий меньше запрошенного: "
                f"{len(selected)} из {count}."
            )

    return 0


async def clear_due_reviews(login: str) -> int:
    async with async_session_factory() as db:
        user = await _find_user(db, login)
        reviews = list(
            (
                await db.execute(
                    select(TrainingReview)
                    .join(TrainingTask, TrainingReview.task_id == TrainingTask.id)
                    .join(TrainingLesson, TrainingTask.lesson_id == TrainingLesson.id)
                    .join(TrainingModule, TrainingLesson.module_id == TrainingModule.id)
                    .where(
                        TrainingReview.user_id == user.id,
                        TrainingTask.enabled.is_(True),
                        TrainingLesson.enabled.is_(True),
                        TrainingModule.enabled.is_(True),
                    )
                )
            ).scalars()
        )

        for review in reviews:
            await db.delete(review)
        await db.commit()

        print(
            f"Пользователь '{login}' (id={user.id}): "
            f"удалено повторений по включённым заданиям: {len(reviews)}."
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Создать тестовые просроченные повторения для пользователя",
    )
    parser.add_argument("login", help="логин пользователя")
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="сколько заданий сделать просроченными (по умолчанию 6)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="на сколько дней назад сдвинуть дату повторения (по умолчанию 1)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="удалить повторения пользователя по включённым заданиям",
    )
    args = parser.parse_args()

    if args.count < 1:
        print("Ошибка: --count должен быть больше нуля.")
        return 2

    if args.clear:
        return asyncio.run(clear_due_reviews(args.login))
    return asyncio.run(seed_due_reviews(args.login, args.count, args.days_back))


if __name__ == "__main__":
    raise SystemExit(main())
