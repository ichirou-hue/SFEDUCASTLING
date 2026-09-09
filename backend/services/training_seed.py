"""Идемпотентное начальное наполнение вкладки «Обучение».

Запуск из корня проекта:
    python -m backend.services.training_seed

Повторный запуск обновляет подготовленные модули/уроки/задачи, не создавая
дубликатов. Ключ задачи внутри seed — (lesson_id, sort_order).
"""

import asyncio

from sqlalchemy import select

from backend.db.session import async_session_factory
from backend.models.training_lesson import TrainingLesson
from backend.models.training_module import TrainingModule
from backend.models.training_task import TrainingTask


MODULES = [
    {
        "slug": "pawn",
        "title": "Пешка",
        "description": "Обычный ход, первый ход на две клетки, взятие и специальные правила.",
        "sort_order": 1,
        "enabled": False,
    },
    {
        "slug": "knight",
        "title": "Конь",
        "description": "Движение буквой «Г», перепрыгивание фигур и взятие конём.",
        "sort_order": 2,
        "enabled": True,
    },
    {
        "slug": "bishop",
        "title": "Слон",
        "description": "Движение по диагоналям, препятствия и взятия.",
        "sort_order": 3,
        "enabled": False,
    },
    {
        "slug": "rook",
        "title": "Ладья",
        "description": "Движение по вертикалям и горизонталям.",
        "sort_order": 4,
        "enabled": False,
    },
    {
        "slug": "queen",
        "title": "Ферзь",
        "description": "Совмещение возможностей ладьи и слона.",
        "sort_order": 5,
        "enabled": False,
    },
    {
        "slug": "king",
        "title": "Король",
        "description": "Движение короля, атакованные поля и безопасность короля.",
        "sort_order": 6,
        "enabled": False,
    },
    {
        "slug": "special-rules",
        "title": "Специальные правила",
        "description": "Рокировка, взятие на проходе и превращение пешки.",
        "sort_order": 7,
        "enabled": False,
    },
    {
        "slug": "check-mate-stalemate",
        "title": "Шах, мат и пат",
        "description": "Базовые окончания партии и способы защиты от шаха.",
        "sort_order": 8,
        "enabled": False,
    },
]


KNIGHT_LESSONS = [
    {
        "slug": "knight-movement",
        "title": "Как ходит конь",
        "theory": (
            "Конь перемещается буквой «Г»: на две клетки по вертикали или горизонтали "
            "и затем на одну клетку перпендикулярно. Из центра доски у коня может быть "
            "до восьми вариантов хода."
        ),
        "sort_order": 1,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь в центре доски",
                "instruction": "Отметьте все клетки, на которые может перейти белый конь с e4.",
                "fen": "7k/8/8/8/4N3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": (
                    "С e4 конь может перейти на c3, c5, d2, d6, f2, f6, g3 и g5. "
                    "Каждый такой ход образует букву «Г»."
                ),
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Первый ход конём",
                "instruction": "Сделайте любой допустимый ход белым конём с g1.",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source_square": "g1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "any_legal_move"},
                "explanation": "В начальной позиции конь g1 может пойти на f3 или h3.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-edge",
        "title": "Конь у края доски",
        "theory": (
            "Чем ближе конь к краю или углу доски, тем меньше клеток ему доступно. "
            "Это одна из причин, по которой конь обычно активнее ближе к центру."
        ),
        "sort_order": 2,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь в углу a1",
                "instruction": "Отметьте все клетки, доступные белому коню с a1.",
                "fen": "7k/8/8/8/8/8/8/N3K3 w - - 0 1",
                "source_square": "a1",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "Из угла a1 коню доступны только b3 и c2.",
                "sort_order": 1,
            },
            {
                "task_type": "select_squares",
                "title": "Конь в углу h8",
                "instruction": "Отметьте все клетки, доступные белому коню с h8.",
                "fen": "k6N/8/8/8/8/8/8/4K3 w - - 0 1",
                "source_square": "h8",
                "difficulty": 1,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "Из угла h8 коню доступны только f7 и g6.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-jump",
        "title": "Перепрыгивание через фигуры",
        "theory": (
            "Конь — единственная шахматная фигура, которая может перепрыгивать через "
            "другие фигуры. Фигуры на соседних клетках не блокируют его ход, но конечная "
            "клетка не может быть занята своей фигурой."
        ),
        "sort_order": 3,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Окружённый конь",
                "instruction": "Конь d4 окружён своими пешками. Отметьте все клетки, на которые он всё равно может перейти.",
                "fen": "7k/8/8/2PPP3/2PNP3/2PPP3/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": (
                    "Соседние пешки не мешают коню: он перепрыгивает через них и ходит "
                    "на b3, b5, c2, c6, e2, e6, f3 или f5."
                ),
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Перепрыгните окружение",
                "instruction": "Сделайте любой допустимый ход конём с d4, не обращая внимания на окружающие пешки.",
                "fen": "7k/8/8/2PPP3/2PNP3/2PPP3/8/K7 w - - 0 1",
                "source_square": "d4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "any_legal_move"},
                "explanation": "Конь может перепрыгнуть через окружающие его фигуры.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-capture",
        "title": "Взятие конём",
        "theory": (
            "Конь берёт фигуру противника на той клетке, на которую приходит своим обычным "
            "ходом буквой «Г». Перепрыгиваемые фигуры при этом не снимаются с доски."
        ),
        "sort_order": 4,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Найдите доступные взятия",
                "instruction": "Отметьте клетки, на которых белый конь e4 может взять фигуру противника.",
                "fen": "7k/8/8/2r3b1/4N3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "capture_squares"},
                "explanation": "Конь e4 может взять ладью на c5 и слона на g5.",
                "sort_order": 1,
            },
            {
                "task_type": "make_move",
                "title": "Выполните взятие",
                "instruction": "Сделайте конём e4 любое доступное взятие.",
                "fen": "7k/8/8/2r3b1/4N3/8/8/K7 w - - 0 1",
                "source_square": "e4",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "legal_capture"},
                "explanation": "Правильным будет Nxc5 или Nxg5: оба хода являются взятием конём.",
                "sort_order": 2,
            },
        ],
    },
    {
        "slug": "knight-review",
        "title": "Закрепление",
        "theory": (
            "В итоговых заданиях вспомните форму хода коня, влияние края доски, "
            "возможность перепрыгивания и правила взятия."
        ),
        "sort_order": 5,
        "tasks": [
            {
                "task_type": "select_squares",
                "title": "Конь после первых ходов",
                "instruction": "Отметьте все легальные клетки для белого коня с g1 в этой позиции.",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "source_square": "g1",
                "difficulty": 2,
                "payload": {"piece": "N", "mode": "all_legal_moves"},
                "explanation": "После хода e2-e4 коню g1 доступны e2, f3 и h3.",
                "sort_order": 1,
            }
        ],
    },
]


async def _upsert_module(session, data: dict) -> TrainingModule:
    module = await session.scalar(select(TrainingModule).where(TrainingModule.slug == data["slug"]))
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

        knight = modules_by_slug["knight"]
        for lesson_data in KNIGHT_LESSONS:
            lesson = await _upsert_lesson(session, knight, lesson_data)
            for task_data in lesson_data["tasks"]:
                await _upsert_task(session, lesson, task_data)

        await session.commit()

    print("[TrainingSeed] Готово: создан/обновлён учебный курс и модуль «Конь».")


if __name__ == "__main__":
    asyncio.run(seed_training())
