"""Идемпотентное наполнение вкладки «Обучение» стандартным курсом.

Запуск из корня проекта:
    python -m backend.services.training_seed

Повторный запуск обновляет подготовленные модули/уроки/задачи и не создаёт
дубликаты. Ключ урока: (module_id, slug), ключ задания: (lesson_id, sort_order).
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

        await session.commit()

    print(
        "[TrainingSeed] Готово: обновлены "
        f"{len(MODULES)} модулей, {lesson_total} уроков и {task_total} заданий."
    )


if __name__ == "__main__":
    asyncio.run(seed_training())
