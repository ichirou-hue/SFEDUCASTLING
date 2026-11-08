"""Synchronize the declared course and fail release on stale public content."""
import asyncio
from backend.db.session import async_session_factory
from backend.services.training_course_data import LESSONS_BY_MODULE, MODULES
from backend.services.training_seed import seed_training
from backend.services.training_service import list_modules


async def main():
    await seed_training()
    async with async_session_factory() as session:
        actual = await list_modules(session)
    expected = {
        module["slug"]: {
            "title": module["title"],
            "lesson_count": len(LESSONS_BY_MODULE[module["slug"]]),
            "task_count": sum(len(lesson["tasks"]) for lesson in LESSONS_BY_MODULE[module["slug"]]),
        } for module in MODULES
    }
    assert {module["slug"] for module in actual} == set(expected), "Course module mismatch"
    for module in actual:
        for key, value in expected[module["slug"]].items():
            assert module[key] == value, f"Course mismatch: {module['slug']} {key}"
    print("Verified declared course:", len(actual), "modules")


if __name__ == "__main__":
    asyncio.run(main())
