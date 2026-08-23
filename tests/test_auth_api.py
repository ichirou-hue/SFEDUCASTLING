"""Тесты регистрации/авторизации (задача 64).

Используют реальную локальную БД (как остальные интеграционные тесты):
каждый тест создаёт пользователя с уникальным логином и подчищает за собой
(refresh_tokens удаляются каскадом по FK ON DELETE CASCADE).
"""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app import app
from backend.db.session import async_session_factory
from backend.models.user import User

client = TestClient(app)

PASSWORD = "Sup3rSecret!"


def _unique(base: str) -> str:
    return f"{base}_{uuid.uuid4().hex[:8]}"


def get_hash_from_db(login: str) -> str:
    async def _get():
        async with async_session_factory() as db:
            u = await db.scalar(select(User).where(User.login == login))
            return u.password_hash

    return asyncio.run(_get())


@pytest.fixture()
def cleanup():
    """Трекает логины и удаляет пользователей после теста."""
    logins: list[str] = []

    def track(login: str) -> str:
        logins.append(login)
        return login

    yield track

    async def _clean():
        if not logins:
            return
        async with async_session_factory() as db:
            await db.execute(delete(User).where(User.login.in_(logins)))
            await db.commit()

    asyncio.run(_clean())


class TestRegister:
    def test_register_ok_and_duplicate(self, cleanup):
        login = cleanup(_unique("user"))
        r = client.post(
            "/api/auth/register", json={"login": login, "password": PASSWORD}
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["user"]["login"] == login
        assert body["user"]["is_admin"] is False
        assert body["access_token"] and body["refresh_token"]
        assert body["token_type"] == "bearer"

        r2 = client.post(
            "/api/auth/register", json={"login": login, "password": PASSWORD}
        )
        assert r2.status_code == 409

    def test_register_short_password(self, cleanup):
        r = client.post(
            "/api/auth/register",
            json={"login": cleanup(_unique("u")), "password": "short"},
        )
        assert r.status_code == 422

    def test_register_bad_login_chars(self):
        r = client.post(
            "/api/auth/register",
            json={"login": "bad login!", "password": PASSWORD},
        )
        assert r.status_code == 422

    def test_password_hashed_bcrypt(self, cleanup):
        login = cleanup(_unique("user"))
        client.post("/api/auth/register", json={"login": login, "password": PASSWORD})
        stored = get_hash_from_db(login)
        assert stored.startswith("$2b$12$")
        assert PASSWORD not in stored


class TestLogin:
    def _register(self, cleanup):
        login = cleanup(_unique("user"))
        email = f"{login}@test.io"
        r = client.post(
            "/api/auth/register",
            json={"login": login, "password": PASSWORD, "email": email},
        )
        assert r.status_code == 201, r.text
        return login, email

    def test_login_by_login(self, cleanup):
        login, _ = self._register(cleanup)
        r = client.post("/api/auth/login", json={"login": login, "password": PASSWORD})
        assert r.status_code == 200, r.text
        assert r.json()["access_token"]

    def test_login_by_email(self, cleanup):
        _, email = self._register(cleanup)
        r = client.post("/api/auth/login", json={"login": email, "password": PASSWORD})
        assert r.status_code == 200, r.text

    def test_login_wrong_password(self, cleanup):
        login, _ = self._register(cleanup)
        r = client.post("/api/auth/login", json={"login": login, "password": "WrongPass1"})
        assert r.status_code == 401

    def test_login_unknown_user(self):
        r = client.post(
            "/api/auth/login", json={"login": _unique("ghost"), "password": PASSWORD}
        )
        assert r.status_code == 401


class TestMeAndTokens:
    def _tokens(self, cleanup):
        login = cleanup(_unique("user"))
        r = client.post("/api/auth/register", json={"login": login, "password": PASSWORD})
        body = r.json()
        return login, body["access_token"], body["refresh_token"]

    def test_me_with_valid_token(self, cleanup):
        login, access, _ = self._tokens(cleanup)
        r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access}"})
        assert r.status_code == 200
        assert r.json()["user"]["login"] == login

    def test_me_without_token(self):
        assert client.get("/api/auth/me").status_code == 401

    def test_me_with_garbage_token(self):
        r = client.get(
            "/api/auth/me", headers={"Authorization": "Bearer garbage.token.here"}
        )
        assert r.status_code == 401

    def test_refresh_rotation_rejects_old_token(self, cleanup):
        _, _, refresh_tok = self._tokens(cleanup)
        r1 = client.post("/api/auth/refresh", json={"refresh_token": refresh_tok})
        assert r1.status_code == 200, r1.text
        new_refresh = r1.json()["refresh_token"]
        assert new_refresh != refresh_tok

        # старый refresh отозван ротацией
        r2 = client.post("/api/auth/refresh", json={"refresh_token": refresh_tok})
        assert r2.status_code == 401
        # новый работает
        r3 = client.post("/api/auth/refresh", json={"refresh_token": new_refresh})
        assert r3.status_code == 200

    def test_logout_revokes_refresh(self, cleanup):
        _, access, refresh_tok = self._tokens(cleanup)
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {access}"})
        assert me.status_code == 200
        r = client.post("/api/auth/logout", json={"refresh_token": refresh_tok})
        assert r.status_code == 200
        assert (
            client.post("/api/auth/refresh", json={"refresh_token": refresh_tok}).status_code
            == 401
        )

    def test_refresh_garbage(self):
        assert (
            client.post("/api/auth/refresh", json={"refresh_token": "x" * 64}).status_code
            == 401
        )


class TestAdmin:
    def test_admin_only_forbidden_for_regular_user(self, cleanup):
        login = cleanup(_unique("user"))
        body = client.post(
            "/api/auth/register", json={"login": login, "password": PASSWORD}
        ).json()
        r = client.get(
            "/api/auth/admin-only",
            headers={"Authorization": f"Bearer {body['access_token']}"},
        )
        assert r.status_code == 403

    def test_admin_account_exists_and_can_login(self):
        """Аккаунт admin существует (создан скриптом) и входит с правами админа."""
        r = client.post(
            "/api/auth/login", json={"login": "admin", "password": "GigaChess2026!"}
        )
        assert r.status_code == 200
        assert r.json()["user"]["is_admin"] is True
