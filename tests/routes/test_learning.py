"""Тесты обучающих endpoint'ов: /api/learning/level-test/*, /api/learning/puzzle/check."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from sqlalchemy import delete, select

from backend.app import app
from backend.db.session import async_session_factory
from backend.models.level_test import LevelTest
from backend.models.user import User

client = TestClient(app)

NO_STOCKFISH = patch("backend.api_gateway.state.ensure_stockfish", return_value=None)

NEW_PUZZLES = {
    "puzzles": [
        # корзина 500-900 (уровень 1)
        {"id": "a1", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 600, "themes": ["mate", "mateIn2"]},
        {"id": "a2", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 700, "themes": ["mate", "mateIn2"]},
        {"id": "a3", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 800, "themes": ["mate", "mateIn2"]},
        {"id": "a4", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 850, "themes": ["fork"]},
        {"id": "a5", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 890, "themes": ["fork"]},
        # корзина 900-1300 (уровень 2)
        {"id": "b1", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 950, "themes": ["fork"]},
        {"id": "b2", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 1050, "themes": ["fork"]},
        {"id": "b3", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 1150, "themes": ["pin"]},
        {"id": "b4", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8 a8a7 b8b7 a7a8 b7b1", "rating": 1250, "themes": ["pin"]},
        {"id": "b5", "fen": "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", "moves": "b1b8", "rating": 1290, "themes": ["hangingPiece"]},
    ]
}

MATCH_ANCHOR = "backend.api_gateway.state.puzzle_base"

PASSWORD = "Sup3rSecret!"

# Минимальная валидная анкета Q1-Q8: без неё /level-test/start отвечает 409.
ONBOARDING = {
    "q1": "sometimes",
    "q3": ["family", "online_rating"],
    "q4": "lt1",
    "q5": ["puzzles", "games", "lessons"],
    "q6": "stuck",
    "q7": "age10_16",
    "q8": "internet",
}


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def cleanup():
    """Удаляет созданных в тесте пользователей вместе с их level-test'ами."""
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
                (
                    await db.scalars(select(User.id).where(User.login.in_(logins)))
                ).all()
            )
            if user_ids:
                await db.execute(
                    delete(LevelTest).where(LevelTest.user_id.in_(user_ids))
                )
            await db.execute(delete(User).where(User.login.in_(logins)))
            await db.commit()

    asyncio.run(_clean())


def _authed_user(cleanup) -> str:
    """Регистрирует пользователя и проходит анкету — условие /level-test/start."""
    login = cleanup(f"lvltest_{uuid.uuid4().hex[:8]}")
    r = client.post(
        "/api/auth/register", json={"login": login, "password": PASSWORD}
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    r = client.post(
        "/api/auth/onboarding", json=ONBOARDING, headers=_headers(token)
    )
    assert r.status_code == 200, r.text
    return token


def _start_test(token: str) -> tuple[int, list[str]]:
    with patch(MATCH_ANCHOR, NEW_PUZZLES):
        resp = client.post("/api/learning/level-test/start", headers=_headers(token))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    return data["test_id"], [q["id"] for q in data["questions"]]


class TestLevelTestStart:
    def test_without_puzzle_base(self, cleanup):
        token = _authed_user(cleanup)
        with patch(MATCH_ANCHOR, None):
            resp = client.post(
                "/api/learning/level-test/start", headers=_headers(token)
            )
        assert resp.status_code == 503
        assert resp.json()["detail"] == "База паззлов не загружена"

    def test_requires_onboarding(self, cleanup):
        login = cleanup(f"lvltest_{uuid.uuid4().hex[:8]}")
        r = client.post(
            "/api/auth/register", json={"login": login, "password": PASSWORD}
        )
        assert r.status_code == 201, r.text
        token = r.json()["access_token"]
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/level-test/start", headers=_headers(token)
            )
        assert resp.status_code == 409
        assert "анкет" in resp.json()["detail"].lower()

    def test_returns_questions_without_solutions(self, cleanup):
        token = _authed_user(cleanup)
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/level-test/start", headers=_headers(token)
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["resumed"] is False
        assert data["total"] == len(data["questions"]) == 10
        ids = [q["id"] for q in data["questions"]]
        assert len(set(ids)) == 10
        assert all("fen" in q and "id" in q for q in data["questions"])
        assert all(
            "moves" not in q and "solution" not in q and "themes" not in q
            for q in data["questions"]
        )
        assert data["personalization"]["target_allocation"]


class TestLevelTestCheck:
    def test_puzzle_base_gone_after_start(self, cleanup):
        token = _authed_user(cleanup)
        test_id, question_ids = _start_test(token)
        with patch(MATCH_ANCHOR, None):
            resp = client.post(
                "/api/learning/level-test/check",
                json={"test_id": test_id, "puzzle_id": question_ids[0], "move": "b1b8"},
                headers=_headers(token),
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Задача не найдена"

    def test_correct_answer(self, cleanup):
        token = _authed_user(cleanup)
        test_id, question_ids = _start_test(token)
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/level-test/check",
                json={"test_id": test_id, "puzzle_id": question_ids[0], "move": "b1b8"},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["correct"] is True
        assert data["recorded"] is True
        # Эталон не выдаётся до submit.
        assert "solution" not in data

    def test_wrong_answer(self, cleanup):
        token = _authed_user(cleanup)
        test_id, question_ids = _start_test(token)
        with NO_STOCKFISH:
            with patch(MATCH_ANCHOR, NEW_PUZZLES):
                resp = client.post(
                    "/api/learning/level-test/check",
                    json={"test_id": test_id, "puzzle_id": question_ids[0], "move": "b1b1"},
                    headers=_headers(token),
                )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["correct"] is False
        assert data["recorded"] is True

    def test_unknown_puzzle(self, cleanup):
        token = _authed_user(cleanup)
        test_id, _ = _start_test(token)
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/level-test/check",
                json={"test_id": test_id, "puzzle_id": "zzz", "move": "b1b8"},
                headers=_headers(token),
            )
        assert resp.status_code == 422
        assert resp.json()["detail"] == "Эта задача не относится к тесту"


class TestLevelTestResult:
    def test_without_puzzle_base(self):
        with patch(MATCH_ANCHOR, None):
            resp = client.post("/api/learning/level-test/result", json=[])
        assert resp.status_code == 200
        assert resp.json().get("error") == "База паззлов не загружена"

    def test_high_level_when_3_of_5_in_top_bucket(self):
        answers = []
        ids = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4", "b5"]
        for i, pid in enumerate(ids):
            answers.append({"puzzle_id": pid, "correct": True})
        # Корзины: 5 задач уровня 1 (все верно), 5 задач уровня 2 (все верно).
        # Наибольший достигнутый уровень, где >=3 из 5: уровень 2.
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post("/api/learning/level-test/result", json=answers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["level"] == 2
        assert data["level"] == 2


class TestPuzzleCheck:
    def test_correct(self):
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/puzzle/check",
                json={"puzzle_id": "b5", "move": "b1b8"},
            )
        assert resp.status_code == 200
        assert resp.json()["correct"] is True
