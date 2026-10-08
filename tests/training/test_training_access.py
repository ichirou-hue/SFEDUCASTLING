"""Доступ к учебным модулям только для зарегистрированных пользователей.

Таска «в модуль можно зайти только зарегистрированным»: эндпоинты
/api/training/* требуют access-токен, гость получает 401.
"""

import uuid

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def _unique(base: str) -> str:
    return f"{base}_{uuid.uuid4().hex[:8]}"


def test_training_modules_requires_auth():
    r = client.get("/api/training/modules")
    assert r.status_code == 401


def test_training_lesson_requires_auth():
    r = client.get("/api/training/lessons/1")
    assert r.status_code == 401


def test_training_task_check_requires_auth():
    r = client.post("/api/training/tasks/1/check", json={"answer": {}})
    assert r.status_code == 401


def test_training_modules_ok_with_token():
    login = _unique("learner")
    reg = client.post(
        "/api/auth/register",
        json={"login": login, "password": "Sup3rSecret!"},
    )
    assert reg.status_code == 201, reg.text
    access = reg.json()["access_token"]
    r = client.get(
        "/api/training/modules",
        headers={"Authorization": f"Bearer {access}"},
    )
    assert r.status_code == 200
    assert "modules" in r.json()