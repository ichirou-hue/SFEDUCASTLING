from datetime import UTC, datetime

import jwt

from backend.api_gateway.security import create_access_token
from backend.config.settings import settings
from backend.models.user import ROLE_ADMIN, ROLE_LEARNER, User


def test_user_default_role_is_learner():
    user = User(login="u", password_hash="x", role=ROLE_LEARNER)
    assert user.effective_role == ROLE_LEARNER
    assert user.public()["role"] == ROLE_LEARNER
    assert user.public()["is_admin"] is False


def test_set_role_changes_single_role_source():
    user = User(login="u", password_hash="x", role=ROLE_LEARNER)
    user.set_role(ROLE_ADMIN)
    assert user.role == ROLE_ADMIN
    assert user.public()["is_admin"] is True
    user.set_role(ROLE_LEARNER)
    assert user.role == ROLE_LEARNER
    assert user.public()["is_admin"] is False


def test_access_token_contains_role_and_long_ttl(monkeypatch):
    monkeypatch.setattr(settings.auth, "access_token_ttl_min", 60 * 24 * 7)
    token = create_access_token(1, "Agro", ROLE_ADMIN)
    payload = jwt.decode(
        token,
        settings.auth.jwt_secret,
        algorithms=["HS256"],
        options={"verify_exp": False},
    )
    assert payload["role"] == ROLE_ADMIN
    assert payload["admin"] is True
    ttl_seconds = payload["exp"] - payload["iat"]
    assert ttl_seconds == 60 * 60 * 24 * 7
    assert payload["exp"] > int(datetime.now(UTC).timestamp())
