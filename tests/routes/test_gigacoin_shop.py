"""GigaCoin shop: покупка, надевание и снятие косметики."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app import app
from backend.db.session import async_session_factory
from backend.models.gigacoin_event import GigacoinEvent
from backend.models.user import User
from backend.models.user_cosmetic import UserCosmetic

client = TestClient(app)

PASSWORD = "Sup3rSecret!"


@pytest.fixture()
def cleanup():
    """Удаляет созданного пользователя вместе с его косметикой и событиями."""
    logins: list[str] = []

    def track(login: str) -> str:
        logins.append(login)
        return login

    yield track

    async def _clean():
        if not logins:
            return
        async with async_session_factory() as db:
            user_ids = list(
                (await db.scalars(select(User.id).where(User.login.in_(logins)))).all()
            )
            if user_ids:
                await db.execute(
                    delete(GigacoinEvent).where(GigacoinEvent.user_id.in_(user_ids))
                )
                await db.execute(
                    delete(UserCosmetic).where(UserCosmetic.user_id.in_(user_ids))
                )
            await db.execute(delete(User).where(User.login.in_(logins)))
            await db.commit()

    asyncio.run(_clean())


def _register(cleanup) -> tuple[str, int]:
    login = cleanup(f"shop_{uuid.uuid4().hex[:8]}")
    r = client.post("/api/auth/register", json={"login": login, "password": PASSWORD})
    assert r.status_code == 201, r.text
    body = r.json()
    return body["access_token"], body["user"]["id"]


def _set_balance(user_id: int, amount: int) -> None:
    async def _set():
        async with async_session_factory() as db:
            user = await db.scalar(select(User).where(User.id == user_id))
            user.gigacoins = amount
            await db.commit()

    asyncio.run(_set())


def _row_equipped(user_id: int, cosmetic_id: str) -> bool:
    async def _get():
        async with async_session_factory() as db:
            row = await db.scalar(
                select(UserCosmetic).where(
                    UserCosmetic.user_id == user_id,
                    UserCosmetic.cosmetic_id == cosmetic_id,
                )
            )
            return bool(row and row.is_equipped)

    return asyncio.run(_get())


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_unequip_clears_active_cosmetic(cleanup):
    token, user_id = _register(cleanup)
    _set_balance(user_id, 500)
    headers = _headers(token)

    r = client.post(
        "/api/gigacoins/shop/purchase",
        json={"cosmetic_id": "frame_bronze"},
        headers=headers,
    )
    assert r.status_code == 200, r.text

    r = client.post(
        "/api/gigacoins/shop/equip",
        json={"cosmetic_id": "frame_bronze"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["user"]["equipped_avatar_frame"] == "frame_bronze"
    assert _row_equipped(user_id, "frame_bronze") is True

    r = client.post(
        "/api/gigacoins/shop/unequip",
        json={"cosmetic_id": "frame_bronze"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cosmetic_id"] == "frame_bronze"
    assert body["slot"] == "avatar_frame"
    assert body["user"]["equipped_avatar_frame"] is None
    assert _row_equipped(user_id, "frame_bronze") is False

    catalog = client.get("/api/gigacoins/shop", headers=headers).json()
    items = {item["id"]: item for item in catalog["items"]}
    assert items["frame_bronze"]["owned"] is True
    assert items["frame_bronze"]["equipped"] is False


def test_unequip_other_item_keeps_active_cosmetic(cleanup):
    token, user_id = _register(cleanup)
    _set_balance(user_id, 500)
    headers = _headers(token)

    for cosmetic_id in ("accent_aurora", "accent_ruby"):
        r = client.post(
            "/api/gigacoins/shop/purchase",
            json={"cosmetic_id": cosmetic_id},
            headers=headers,
        )
        assert r.status_code == 200, r.text

    r = client.post(
        "/api/gigacoins/shop/equip",
        json={"cosmetic_id": "accent_aurora"},
        headers=headers,
    )
    assert r.status_code == 200, r.text

    r = client.post(
        "/api/gigacoins/shop/unequip",
        json={"cosmetic_id": "accent_ruby"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["user"]["equipped_profile_accent"] == "accent_aurora"
    assert _row_equipped(user_id, "accent_aurora") is True


def test_unequip_unknown_item_returns_404(cleanup):
    token, _ = _register(cleanup)
    r = client.post(
        "/api/gigacoins/shop/unequip",
        json={"cosmetic_id": "missing_item"},
        headers=_headers(token),
    )
    assert r.status_code == 404
