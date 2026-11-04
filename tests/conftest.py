"""Test-only setup shared by the backend test suite."""

import asyncio
import os

import pytest
from sqlalchemy import select

# Стартовый прогрев Maia3 поднимает настоящий UCI-процесс (torch,
# чекпоинт). Тестам он не нужен и только замедляет прогон.
os.environ.setdefault("MAIA3_PRELOAD", "0")


@pytest.fixture(scope="session", autouse=True)
def ensure_ci_admin_account():
    """Seed the admin expected by API tests in the disposable CI database."""
    if os.environ.get("CI", "").lower() != "true":
        return

    from backend.api_gateway.security import hash_password
    from backend.db.session import async_session_factory
    from backend.models.user import ROLE_ADMIN, User

    async def seed_admin():
        async with async_session_factory() as db:
            admin = await db.scalar(select(User).where(User.login == "admin"))
            if admin is None:
                admin = User(login="admin", role=ROLE_ADMIN)
                db.add(admin)
            admin.role = ROLE_ADMIN
            admin.password_hash = hash_password("GigaChess2026!")
            await db.commit()

    asyncio.run(seed_admin())
