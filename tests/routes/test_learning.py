"""Тесты обучающих endpoint'ов: /api/learning/level-test/*, /api/learning/puzzle/check."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from sqlalchemy import delete, select

from backend.app import app
from backend.db.session import async_session_factory
from backend.models.level_test import LevelTest
from backend.models.puzzle_attempt import PuzzleAttempt
from backend.models.user import User
from backend.models.user_chess_account import UserChessAccount

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

# Валидная позиция и решение для проверки отбора по рейтингу.
RATING_FEN = "k7/8/8/8/8/8/8/1R4K1 w - - 0 1"
RATING_MOVES = "b1b8 a8a7 b8b7 a7a8 b7b1"
RATING_PUZZLES = {
    "puzzles": [
        # группа 1 (0-1000)
        {"id": "low1", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 500, "themes": ["fork"]},
        {"id": "low2", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 900, "themes": ["fork"]},
        # группа 2 (1001-1500) — своя для рейтинга 1200
        {"id": "mid1", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1200, "themes": ["pin"]},
        {"id": "mid2", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1350, "themes": ["pin"]},
        {"id": "mid3", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1400, "themes": ["fork"]},
        # группа 3 (1501-1900) — соседняя, но выше потолка 1200+200=1400
        {"id": "high1", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1600, "themes": ["sacrifice"]},
        {"id": "high2", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1800, "themes": ["sacrifice"]},
        # группа 4 (1901+) — отсекается и диапазоном, и потолком
        {"id": "top1", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 2100, "themes": ["sacrifice"]},
    ]
}

# Пазлы модуля «Фигуры» (pieces): тема rookEndgame входит в этот модуль,
# поэтому они проходят проверку темы курса.
TOPIC_PUZZLES = {
    "puzzles": [
        {"id": "rook_low", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 500, "themes": ["rookEndgame"]},
        {"id": "rook_mid", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1200, "themes": ["rookEndgame"]},
        {"id": "rook_high", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1600, "themes": ["rookEndgame"]},
        {"id": "rook_top", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 2100, "themes": ["rookEndgame"]},
    ]
}

# Пробный режим до анкеты: пять задач с рейтингом до 700 и две выше границы.
PREVIEW_PUZZLES = {
    "puzzles": [
        {"id": "p1", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 400, "themes": ["fork"]},
        {"id": "p2", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 500, "themes": ["fork"]},
        {"id": "p3", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 600, "themes": ["fork"]},
        {"id": "p4", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 650, "themes": ["fork"]},
        {"id": "p5", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 700, "themes": ["fork"]},
        {"id": "p_high1", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1200, "themes": ["fork"]},
        {"id": "p_high2", "fen": RATING_FEN, "moves": RATING_MOVES, "rating": 1500, "themes": ["fork"]},
    ]
}

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
                await db.execute(
                    delete(PuzzleAttempt).where(
                        PuzzleAttempt.user_id.in_(user_ids)
                    )
                )
                await db.execute(
                    delete(UserChessAccount).where(
                        UserChessAccount.user_id.in_(user_ids)
                    )
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


def _authed_user_with_rating(cleanup, rating: int) -> str:
    """Пользователь с привязанным внешним рейтингом.

    Только привязка (rating_usable=True) снимает ограничение самооценки 800,
    поэтому стартовая оценка берётся из внешнего рейтинга.
    """
    login = cleanup(f"lvlrated_{uuid.uuid4().hex[:8]}")
    r = client.post(
        "/api/auth/register", json={"login": login, "password": PASSWORD}
    )
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    user_id = r.json()["user"]["id"]

    async def _link():
        async with async_session_factory() as db:
            db.add(
                UserChessAccount(
                    user_id=user_id,
                    platform="lichess",
                    username="rated",
                    rating_type="blitz",
                    rating=rating,
                    rating_scale="lichess_blitz",
                    games=200,
                    rating_usable=True,
                )
            )
            await db.commit()

    asyncio.run(_link())
    r = client.post(
        "/api/auth/onboarding", json=ONBOARDING, headers=_headers(token)
    )
    assert r.status_code == 200, r.text
    return token


def _registered_user(cleanup) -> tuple[str, int]:
    """Регистрирует пользователя, но НЕ проходит анкету (пробный режим)."""
    login = cleanup(f"lvlnonboard_{uuid.uuid4().hex[:8]}")
    r = client.post(
        "/api/auth/register", json={"login": login, "password": PASSWORD}
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["access_token"], body["user"]["id"]


def _add_puzzle_attempts(
    user_id: int, puzzle_ids: list[str], *, days_ago: int = 0
) -> None:
    async def _run() -> None:
        async with async_session_factory() as db:
            created_at = None
            if days_ago:
                # naive-UTC «сейчас» — так же, как пишет приложение
                # (created_at — timestamp без tz, серверный CURRENT_TIMESTAMP).
                # Считаем в Python, а не SQL: CAST(CURRENT_TIMESTAMP AS DATETIME)
                # в SQLite даёт 0 (NUMERIC-аффинитет), и драйвер падает.
                created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(
                    days=days_ago
                )
            for pid in puzzle_ids:
                values: dict = {"user_id": user_id, "puzzle_id": pid, "correct": True}
                if created_at is not None:
                    values["created_at"] = created_at
                db.add(PuzzleAttempt(**values))
            await db.commit()

    asyncio.run(_run())


def _start_test(token: str) -> tuple[int, list[str]]:
    with patch(MATCH_ANCHOR, NEW_PUZZLES):
        resp = client.post("/api/learning/level-test/start", headers=_headers(token))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    return data["test_id"], [q["id"] for q in data["questions"]]


class TestGigaCoinLearningRewards:
    def test_puzzle_reward_is_added_once(self, cleanup):
        token, _ = _registered_user(cleanup)
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            first = client.post(
                "/api/learning/puzzle/attempt",
                json={"puzzle_id": "b5", "correct": True, "player_moves": ["b1b8"]},
                headers=_headers(token),
            )
            second = client.post(
                "/api/learning/puzzle/attempt",
                json={"puzzle_id": "b5", "correct": True, "player_moves": ["b1b8"]},
                headers=_headers(token),
            )
        assert first.status_code == second.status_code == 200
        assert first.json()["gigacoins_earned"] == 2  # rating 1290
        assert first.json()["gigacoins_balance"] == 2
        assert second.json()["gigacoins_earned"] == 0
        assert second.json()["gigacoins_balance"] == 2

    def test_onboarding_rewards_once_even_when_resubmitted(self, cleanup):
        token, _ = _registered_user(cleanup)
        first = client.post(
            "/api/auth/onboarding", json=ONBOARDING, headers=_headers(token)
        )
        second = client.post(
            "/api/auth/onboarding", json=ONBOARDING, headers=_headers(token)
        )
        assert first.status_code == second.status_code == 200
        assert first.json()["gigacoins_earned"] == 5
        assert first.json()["gigacoins_balance"] == 5
        assert second.json()["gigacoins_earned"] == 0
        assert second.json()["gigacoins_balance"] == 5

    def test_initial_level_test_rewards_once(self, cleanup):
        token, _ = _registered_user(cleanup)
        onboarding = client.post(
            "/api/auth/onboarding", json=ONBOARDING, headers=_headers(token)
        )
        assert onboarding.status_code == 200

        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            test_id, question_ids = _start_test(token)
            payload = {
                "test_id": test_id,
                "answers": [
                    {"puzzle_id": puzzle_id, "correct": False}
                    for puzzle_id in question_ids
                ],
            }
            first = client.post(
                "/api/learning/level-test/submit",
                json=payload,
                headers=_headers(token),
            )
            second = client.post(
                "/api/learning/level-test/submit",
                json=payload,
                headers=_headers(token),
            )
        assert first.status_code == second.status_code == 200
        assert first.json()["gigacoins_earned"] == 10
        assert first.json()["gigacoins_balance"] == 15
        assert second.json()["gigacoins_earned"] == 0
        assert second.json()["gigacoins_balance"] == 15


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


class TestPuzzleSelectionByRating:
    def test_guest_gets_uniform_buckets(self):
        with patch(MATCH_ANCHOR, RATING_PUZZLES):
            resp = client.get("/api/learning/puzzles", params={"count": 8})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 8
        assert data["personalization"] == {
            "source": "none",
            "group": None,
            "max_rating": None,
        }

    def test_rated_user_gets_only_own_group(self, cleanup):
        token = _authed_user_with_rating(cleanup, 1200)
        with patch(MATCH_ANCHOR, RATING_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 5},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # Привязанный рейтинг 1200 -> группа 2, потолок 1400.
        # Только своя корзина: соседняя (low) и выше потолка (high) не выдаются.
        assert data["personalization"] == {
            "source": "onboarding",
            "group": 2,
            "max_rating": 1400,
        }
        ids = {p["id"] for p in data["puzzles"]}
        assert ids == {"mid1", "mid2", "mid3"}
        assert "low1" not in ids and "low2" not in ids
        assert data["total"] == len(ids) == 3

    def test_puzzles_above_rating_plus_200_are_never_offered(self, cleanup):
        token = _authed_user_with_rating(cleanup, 1200)
        # Просим заведомо больше, чем есть: потолок должен держать список.
        with patch(MATCH_ANCHOR, RATING_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 50},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["personalization"]["max_rating"] == 1400
        assert all(p["rating"] <= 1400 for p in data["puzzles"])
        ids = {p["id"] for p in data["puzzles"]}
        # Ни выше потолка, ни из чужой (нижней) корзины.
        assert "high1" not in ids and "high2" not in ids and "top1" not in ids
        assert "low1" not in ids and "low2" not in ids

    def test_topic_practice_ignores_rating(self, cleanup):
        # Практика по теме модуля не привязана к рейтингу: доступны все пазлы
        # темы, включая сильно ниже и выше корзины пользователя.
        token = _authed_user_with_rating(cleanup, 1200)
        with patch(MATCH_ANCHOR, TOPIC_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 20, "topic": "pieces"},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["personalization"] == {
            "source": None,
            "group": None,
            "max_rating": None,
        }
        ids = {p["id"] for p in data["puzzles"]}
        assert ids == {"rook_low", "rook_mid", "rook_high", "rook_top"}


class TestAdaptivePuzzleSelectionByRating:
    def test_adaptive_caps_to_own_group(self, cleanup):
        # Персональная тренировка тоже ограничена текущей корзиной (+200).
        token = _authed_user_with_rating(cleanup, 1200)
        with patch(MATCH_ANCHOR, RATING_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles/adaptive",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["max_puzzle_rating"] == 1400
        assert data["total"] > 0
        for p in data["puzzles"]:
            assert 1001 <= p["rating"] <= 1400
        ids = {p["id"] for p in data["puzzles"]}
        assert "low1" not in ids and "low2" not in ids
        assert "high1" not in ids and "top1" not in ids


class TestPuzzlePreviewBeforeOnboarding:
    def test_preview_limits_to_three_low_puzzles(self, cleanup):
        # Зарегистрирован, но анкету не прошёл: не больше 3 задач до 700.
        token, _uid = _registered_user(cleanup)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["preview"] is True
        assert data["preview_limit"] == 3
        assert data["preview_remaining"] == 3
        assert data["preview_locked"] is False
        assert data["total"] == 3
        assert len(data["puzzles"]) == 3
        assert all(p["rating"] <= 700 for p in data["puzzles"])

    def test_preview_locks_after_three_distinct_attempts(self, cleanup):
        token, user_id = _registered_user(cleanup)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            first = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            )
        ids = [p["id"] for p in first.json()["puzzles"]]
        assert len(ids) == 3
        _add_puzzle_attempts(user_id, ids)

        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["preview"] is True
        assert data["preview_remaining"] == 0
        assert data["preview_locked"] is True
        assert data["puzzles"] == []
        assert data["total"] == 0

    def test_preview_resets_next_day(self, cleanup):
        # Лимит — на сутки: вчерашние попытки не тратят сегодняшний запас.
        token, user_id = _registered_user(cleanup)
        _add_puzzle_attempts(user_id, ["old1", "old2", "old3"], days_ago=1)
        _add_puzzle_attempts(user_id, ["old4"], days_ago=2)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["preview"] is True
        assert data["preview_remaining"] == 3
        assert data["preview_locked"] is False
        assert data["total"] == 3

    def test_preview_partial_day_usage(self, cleanup):
        # Одна задача сегодня → остаётся две.
        token, user_id = _registered_user(cleanup)
        _add_puzzle_attempts(user_id, ["today1"])
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["preview_remaining"] == 2
        assert data["preview_locked"] is False
        assert data["total"] == 2

    def test_preview_assignment_is_stable_across_reloads(self, cleanup):
        # Повторное открытие вкладки «Шахматные задачи» не меняет набор.
        token, _uid = _registered_user(cleanup)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            first = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            ).json()
            second = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            ).json()
        assert len(first["puzzles"]) == 3
        assert [p["id"] for p in first["puzzles"]] == [
            p["id"] for p in second["puzzles"]
        ]

    def test_preview_last_puzzle_stays_fixed(self, cleanup):
        # Решив две задачи, при перезапуске вкладки получаем ту же третью задачу.
        token, user_id = _registered_user(cleanup)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            batch = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            ).json()
        ids = [p["id"] for p in batch["puzzles"]]
        assert len(ids) == 3
        _add_puzzle_attempts(user_id, ids[:2])

        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            after = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            ).json()
        assert after["preview_remaining"] == 1
        assert after["total"] == 1
        assert after["puzzles"][0]["id"] == ids[2]

        # И ещё один перезапуск отдаёт ровно ту же (последнюю) задачу.
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            again = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            ).json()
        assert [p["id"] for p in again["puzzles"]] == [ids[2]]

    def test_preview_adaptive_caps_to_three(self, cleanup):
        token, _uid = _registered_user(cleanup)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles/adaptive",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["preview"] is True
        assert data["preview_limit"] == 3
        assert data["total"] <= 3
        assert all(p["rating"] <= 700 for p in data["puzzles"])

    def test_onboarded_user_is_not_preview(self, cleanup):
        # Прошедший анкету получает обычную персональную выборку.
        token = _authed_user_with_rating(cleanup, 1200)
        with patch(MATCH_ANCHOR, PREVIEW_PUZZLES):
            resp = client.get(
                "/api/learning/puzzles",
                params={"count": 20},
                headers=_headers(token),
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data.get("preview") is None
        assert data["personalization"]["group"] == 2


class TestLearningProgressAccuracy:
    """`accuracy` — solve rate (решённые/взятые по уникальным puzzle_id)."""

    def _register(self) -> tuple[str, int]:
        login = f"lvlacc_{uuid.uuid4().hex[:8]}"
        r = client.post(
            "/api/auth/register", json={"login": login, "password": PASSWORD}
        )
        assert r.status_code == 201, r.text
        body = r.json()
        return body["access_token"], body["user"]["id"]

    def _cleanup(self, user_id: int) -> None:
        async def _clean():
            async with async_session_factory() as db:
                await db.execute(
                    delete(PuzzleAttempt).where(PuzzleAttempt.user_id == user_id)
                )
                await db.execute(delete(User).where(User.id == user_id))
                await db.commit()

        asyncio.run(_clean())

    def test_accuracy_is_solve_rate_not_move_ratio(self):
        token, user_id = self._register()

        async def _seed():
            async with async_session_factory() as db:
                db.add_all(
                    [
                        # p1: два неверных хода, затем решение.
                        PuzzleAttempt(user_id=user_id, puzzle_id="p1", correct=False),
                        PuzzleAttempt(user_id=user_id, puzzle_id="p1", correct=False),
                        PuzzleAttempt(user_id=user_id, puzzle_id="p1", correct=True),
                        # p2 решён с первого хода.
                        PuzzleAttempt(user_id=user_id, puzzle_id="p2", correct=True),
                    ]
                )
                await db.commit()

        try:
            asyncio.run(_seed())
            resp = client.get("/api/learning/progress", headers=_headers(token))
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["attempted"] == 2
            assert data["solved"] == 2
            assert data["attempts"] == 4
            assert data["correct_attempts"] == 2
            # Обе задачи решены -> 100%, а не 2/4 по ходам.
            assert data["accuracy"] == 100.0
        finally:
            self._cleanup(user_id)

    def test_accuracy_is_none_without_attempts(self):
        token, user_id = self._register()
        try:
            resp = client.get("/api/learning/progress", headers=_headers(token))
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["attempted"] == 0
            assert data["solved"] == 0
            assert data["accuracy"] is None
        finally:
            self._cleanup(user_id)
