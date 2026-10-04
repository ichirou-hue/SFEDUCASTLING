"""Разделение прав доступа: guest / learner / admin (задача про роли).

- guest  (без токена): /api/admin/* → 401;
- learner (роль по умолчанию): /api/admin/* → 403, свои данные видит;
- admin: /api/admin/* → 200, видит список пользователей.
"""

import uuid

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)

ADMIN_LOGIN = "admin"
ADMIN_PASSWORD = "GigaChess2026!"


def _unique(base: str) -> str:
    return f"{base}_{uuid.uuid4().hex[:8]}"


def _learner_token() -> str:
    login = _unique("learner")
    r = client.post(
        "/api/auth/register",
        json={"login": login, "password": "Sup3rSecret!"},
    )
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def _create_learner() -> dict:
    login = _unique("learner")
    r = client.post(
        "/api/auth/register",
        json={"login": login, "password": "Sup3rSecret!"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _admin_token() -> str:
    r = client.post(
        "/api/auth/login",
        json={"login": ADMIN_LOGIN, "password": ADMIN_PASSWORD},
    )
    assert r.status_code == 200, r.text
    assert r.json()["user"]["role"] == "admin"
    return r.json()["access_token"]


def test_admin_users_requires_auth():
    assert client.get("/api/admin/users").status_code == 401


def test_admin_users_forbidden_for_learner():
    token = _learner_token()
    r = client.get(
        "/api/admin/users", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403
    assert "Недостаточно прав" in r.json()["detail"]


def test_admin_users_ok_for_admin():
    token = _admin_token()
    r = client.get(
        "/api/admin/users", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "users" in body
    logins = [u["login"] for u in body["users"]]
    assert ADMIN_LOGIN in logins
    sample = body["users"][0]
    assert set(sample) == {
        "id",
        "login",
        "role",
        "is_admin",
        "elo",
        "skill_band",
        "created_at",
    }


def test_admin_only_ok_for_admin():
    token = _admin_token()
    r = client.get(
        "/api/auth/admin-only", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200


def test_learner_me_has_role():
    token = _learner_token()
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["user"]["role"] == "learner"


def test_admin_user_profile_ok_for_admin():
    created = _create_learner()
    learner_id = created["user"]["id"]
    token = _admin_token()
    r = client.get(
        f"/api/admin/users/{learner_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()["user"]
    assert body["id"] == learner_id
    assert body["login"] == created["user"]["login"]
    assert body["role"] == "learner"
    assert "password" not in r.text


def test_admin_user_stats_ok_for_admin():
    created = _create_learner()
    learner_id = created["user"]["id"]
    token = _admin_token()
    r = client.get(
        f"/api/admin/users/{learner_id}/stats",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"user", "training", "learning", "weaknesses"}
    assert set(body["training"]) == {"modules", "attempts", "streak"}
    assert set(body["learning"]) == {
        "attempted",
        "solved",
        "attempts",
        "correct_attempts",
        "accuracy",
    }
    assert body["training"]["modules"]["total"] > 0


def test_admin_user_profile_denied_for_guest():
    created = _create_learner()
    r = client.get(f"/api/admin/users/{created['user']['id']}")
    assert r.status_code == 401


def test_admin_user_profile_denied_for_learner():
    created = _create_learner()
    token = _learner_token()
    r = client.get(
        f"/api/admin/users/{created['user']['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_admin_user_profile_not_found():
    token = _admin_token()
    r = client.get(
        "/api/admin/users/999999999",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "Пользователь не найден"