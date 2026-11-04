"""Общие fixtures для backend-тестов.

Здесь собраны изолированные пользователи и очистка всех связанных строк,
чтобы тесты не зависели друг от друга и не оставляли мусор в БД.
"""

from __future__ import annotations

import asyncio
import os
import uuid

# Лимитер частоты по умолчанию выключен в тестах: десятки тестов ходят в
# /api/auth/* с одного testclient-адреса. Тесты самого лимитера включают его
# точечно через monkeypatch.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app import app
from backend.db.session import async_session_factory
from backend.models.chat_message import ChatMessage
from backend.models.dataset_move import DatasetMove
from backend.models.game import Game, GameMove
from backend.models.gigacoin_event import GigacoinEvent
from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.training_attempt import TrainingAttempt
from backend.models.training_review import TrainingReview
from backend.models.user import RefreshToken, User
from backend.models.user_achievement import UserAchievement
from backend.models.user_activity_day import UserActivityDay
from backend.models.user_chess_account import ChessAccountVerification, UserChessAccount
from backend.models.user_cosmetic import UserCosmetic
from backend.models.user_theme_difficulty import UserThemeDifficulty

PASSWORD = "Sup3rSecret!"


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def ensure_ci_admin_account():
    """Seed the admin expected by API tests in the disposable CI database."""
    if os.environ.get("CI", "").lower() != "true":
        return

    from backend.api_gateway.security import hash_password
    from backend.models.user import ROLE_ADMIN

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


async def _delete_test_users(logins: list[str]) -> None:
    async with async_session_factory() as db:
        user_ids = list(
            (await db.scalars(select(User.id).where(User.login.in_(logins)))).all()
        )
        if not user_ids:
            return

        game_ids = list(
            (await db.scalars(select(Game.id).where(Game.user_id.in_(user_ids)))).all()
        )
        if game_ids:
            await db.execute(delete(GameMove).where(GameMove.game_id.in_(game_ids)))

        account_ids = list(
            (
                await db.scalars(
                    select(UserChessAccount.id).where(
                        UserChessAccount.user_id.in_(user_ids)
                    )
                )
            ).all()
        )
        if account_ids:
            await db.execute(
                delete(ChessAccountVerification).where(
                    ChessAccountVerification.chess_account_id.in_(account_ids)
                )
            )
        await db.execute(
            delete(ChessAccountVerification).where(
                ChessAccountVerification.user_id.in_(user_ids)
            )
        )

        # SQLite не включает foreign keys по умолчанию, поэтому чистим
        # зависимые строки явно, не полагаясь на ON DELETE CASCADE.
        for model in (
            UserAchievement,
            UserActivityDay,
            UserCosmetic,
            UserThemeDifficulty,
            GigacoinEvent,
            PuzzleAttempt,
            LevelTest,
            TrainingAttempt,
            TrainingReview,
            ChatMessage,
            DatasetMove,
            RefreshToken,
            UserChessAccount,
        ):
            await db.execute(delete(model).where(model.user_id.in_(user_ids)))

        await db.execute(delete(Game).where(Game.user_id.in_(user_ids)))
        await db.execute(delete(User).where(User.id.in_(user_ids)))
        await db.commit()


@pytest.fixture()
def user_factory(client: TestClient):
    """Создаёт пользователя через API и удаляет его после теста.

    Возвращает словарь с login, password, access_token, refresh_token,
    user и готовым заголовком Authorization.
    """
    created: list[str] = []

    def _create(prefix: str = "user", *, role: str = "learner") -> dict:
        login = f"{prefix}_{uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/auth/register",
            json={"login": login, "password": PASSWORD},
        )
        assert response.status_code == 201, response.text
        body = response.json()

        if role != "learner":
            async def _set_role() -> None:
                async with async_session_factory() as db:
                    user = await db.scalar(select(User).where(User.login == login))
                    assert user is not None
                    user.role = role
                    await db.commit()

            asyncio.run(_set_role())

        created.append(login)
        return {
            "login": login,
            "password": PASSWORD,
            "access_token": body["access_token"],
            "refresh_token": body["refresh_token"],
            "user": body["user"],
            "headers": {"Authorization": f"Bearer {body['access_token']}"},
        }

    yield _create

    if created:
        asyncio.run(_delete_test_users(created))


@pytest.fixture()
def auth_headers(user_factory) -> dict[str, str]:
    """Заголовок Authorization обычного пользователя."""
    return user_factory("auth")["headers"]


@pytest.fixture()
def admin_headers(user_factory) -> dict[str, str]:
    """Заголовок Authorization администратора."""
    return user_factory("admin", role="admin")["headers"]
