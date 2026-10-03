"""Инициализация базы данных (кросс-платформенная: Windows/Linux/macOS).

Применяет все миграции Alembic к БД из DATABASE_URL и при желании
создаёт администратора.

Использование (из корня проекта, с активированным venv):

    python -m scripts.init_db                         # только схема
    python -m scripts.init_db --admin admin Parol2026  # + админ
    python -m scripts.init_db --check                 # показать текущую ревизию

Без .env используется локальный SQLite (см. .env.example).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

BASE_DIR = Path(__file__).resolve().parent.parent


def _alembic_config() -> Config:
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BASE_DIR / "alembic"))
    return cfg


async def _fetch_revisions(url: str) -> list[str]:
    """Читаем alembic_version напрямую — работает и для aiosqlite, и для asyncpg."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            rows = await conn.execute(text("SELECT version_num FROM alembic_version"))
            return [row[0] for row in rows]
    except Exception:
        return []
    finally:
        await engine.dispose()


def current_revisions() -> list[str]:
    """Текущие ревизии (пусто — таблицы alembic_version нет или она пуста)."""
    from backend.config.settings import settings

    return asyncio.run(_fetch_revisions(settings.database.url))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Создать/обновить схему БД")
    parser.add_argument(
        "--admin",
        nargs=2,
        metavar=("LOGIN", "PASSWORD"),
        help="создать администратора после миграций",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="только показать текущую ревизию и выйти",
    )
    args = parser.parse_args(argv)

    # Гарантируем, что backend.* импортируется из корня проекта.
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    cfg = _alembic_config()
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()

    if args.check:
        current = current_revisions()
        up_to_date = current == [head]
        print(f"Текущая ревизия: {', '.join(current) if current else 'нет данных'}")
        print(f"Head:            {head}")
        print("Схема актуальна." if up_to_date else "Требуется: python -m scripts.init_db")
        return 0 if up_to_date else 1

    from backend.config.settings import settings

    print(f"База: {settings.database.url}")
    command.upgrade(cfg, "head")
    print(f"Схема обновлена до ревизии {head}.")

    if args.admin:
        from scripts.create_admin import create_admin

        login, password = args.admin
        if len(login) < 3 or len(login) > 32:
            print("Ошибка: логин должен быть 3-32 символа", file=sys.stderr)
            return 1
        if len(password) < 8:
            print("Ошибка: пароль минимум 8 символов", file=sys.stderr)
            return 1
        asyncio.run(create_admin(login, password, None, set_password=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
