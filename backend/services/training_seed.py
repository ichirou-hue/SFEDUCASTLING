"""Идемпотентное наполнение вкладки «Обучение» стандартным курсом.

Запуск из корня проекта:
    python -m backend.services.training_seed

Повторный запуск обновляет подготовленные модули/уроки/задачи и не создаёт
дубликаты. Ключ урока: (module_id, slug), ключ задания: (lesson_id, sort_order).

При слиянии модулей уроки переезжают по slug с сохранением id
(иначе оборвутся training_attempts); опустевшие старые модули сносятся.
"""

import asyncio

from sqlalchemy import select

from backend.db.session import async_session_factory
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask
from backend.services.training_course_data import LESSONS_BY_MODULE, MODULES


async def _upsert_module(session, data: dict) -> TrainingModule:
    module = await session.scalar(
        select(TrainingModule).where(TrainingModule.slug == data["slug"])
    )
    if module is None:
        module = TrainingModule(**data)
        session.add(module)
        await session.flush()
        return module

    for field, value in data.items():
        setattr(module, field, value)
    await session.flush()
    return module


async def _upsert_lesson(session, module: TrainingModule, data: dict) -> TrainingLesson:
    lesson = await session.scalar(
        select(TrainingLesson).where(
            TrainingLesson.module_id == module.id,
            TrainingLesson.slug == data["slug"],
        )
    )
    values = {k: v for k, v in data.items() if k != "tasks"}

    if lesson is None:
        # Переезд при слиянии модулей: урок с таким slug уже может лежать
        # в другом (старом) модуле. Переносим его с сохранением id -
        # иначе оборвутся training_attempts и сбросится прогресс.
        lesson = await session.scalar(
            select(TrainingLesson).where(
                TrainingLesson.slug == data["slug"],
            )
        )
        if lesson is not None and lesson.module_id != module.id:
            old_module_id = lesson.module_id
            lesson.module_id = module.id
            for field, value in values.items():
                setattr(lesson, field, value)
            lesson.enabled = True
            await session.flush()
            print(
                "[TrainingSeed] Урок "
                f"{data['slug']}: переезд из модуля id={old_module_id} "
                f"в {module.slug} (id={module.id}), id урока сохранён."
            )
            return lesson

    if lesson is None:
        lesson = TrainingLesson(module_id=module.id, enabled=True, **values)
        session.add(lesson)
        await session.flush()
    else:
        for field, value in values.items():
            setattr(lesson, field, value)
        lesson.enabled = True
        await session.flush()

    return lesson


async def _upsert_task(session, lesson: TrainingLesson, data: dict) -> TrainingTask:
    task = await session.scalar(
        select(TrainingTask).where(
            TrainingTask.lesson_id == lesson.id,
            TrainingTask.sort_order == data["sort_order"],
        )
    )

    if task is None:
        task = TrainingTask(lesson_id=lesson.id, enabled=True, **data)
        session.add(task)
        await session.flush()
        return task

    for field, value in data.items():
        setattr(task, field, value)
    task.enabled = True
    await session.flush()
    return task


async def seed_training() -> None:
    async with async_session_factory() as session:
        modules_by_slug: dict[str, TrainingModule] = {}

        for module_data in MODULES:
            module = await _upsert_module(session, module_data)
            modules_by_slug[module.slug] = module

        lesson_total = 0
        task_total = 0

        for module_slug, lessons in LESSONS_BY_MODULE.items():
            module = modules_by_slug[module_slug]
            for lesson_data in lessons:
                lesson = await _upsert_lesson(session, module, lesson_data)
                lesson_total += 1

                for task_data in lesson_data["tasks"]:
                    await _upsert_task(session, lesson, task_data)
                    task_total += 1

                # Задания, которых больше нет в декларативных данных (например,
                # урок дебюта переписан под развилки и старые sort_order
                # исчезли), гасим, а не удаляем: попытки и повторения на них
                # остаются в истории, но в уроке они не показываются.
                keep_orders = {
                    task_data["sort_order"] for task_data in lesson_data["tasks"]
                }
                stale_tasks = (
                    await session.scalars(
                        select(TrainingTask).where(
                            TrainingTask.lesson_id == lesson.id,
                            TrainingTask.sort_order.notin_(keep_orders),
                            TrainingTask.enabled.is_(True),
                        )
                    )
                ).all()
                for stale_task in stale_tasks:
                    stale_task.enabled = False

            # Уроки, которых больше нет в программе модуля, скрываем, а не
            # удаляем: на них могут ссылаться training_attempts. Прогресс по
            # совпавшим slug сохраняется, потому что урок обновляется на месте.
            keep_lesson_slugs = {
                lesson_data["slug"] for lesson_data in lessons
            }
            stale_lessons = (
                await session.scalars(
                    select(TrainingLesson).where(
                        TrainingLesson.module_id == module.id,
                        TrainingLesson.slug.notin_(keep_lesson_slugs),
                        TrainingLesson.enabled.is_(True),
                    )
                )
            ).all()
            for stale_lesson in stale_lessons:
                stale_lesson.enabled = False
                print(
                    "[TrainingSeed] Урок "
                    f"{stale_lesson.slug} скрыт: его нет в новой программе."
                )

        # Снос опустевших старых модулей после слияния (10 -> 5).
        # Удаляем только модули вне нового состава И без уроков:
        # уроки уехали выше с сохранением id, попытки не страдают.
        # Модули с чужими (не из сида) уроками не трогаем.
        new_slugs = {data["slug"] for data in MODULES}
        stale_modules = (
            await session.scalars(
                select(TrainingModule).where(
                    TrainingModule.slug.notin_(new_slugs)
                )
            )
        ).all()
        removed = 0
        for stale in stale_modules:
            leftover = await session.scalar(
                select(TrainingLesson.id).where(
                    TrainingLesson.module_id == stale.id
                )
            )
            if leftover is None:
                await session.delete(stale)
                removed += 1
                print(f"[TrainingSeed] Удалён пустой старый модуль {stale.slug}.")
            else:
                print(
                    "[TrainingSeed] Модуль "
                    f"{stale.slug} не тронут: в нём остались уроки."
                )

        await session.commit()

    print(
        "[TrainingSeed] Готово: обновлены "
        f"{len(MODULES)} модулей, {lesson_total} уроков и {task_total} заданий."
    )


if __name__ == "__main__":
    asyncio.run(seed_training())
