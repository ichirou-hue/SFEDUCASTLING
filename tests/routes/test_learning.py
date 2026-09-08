"""Тесты обучающих endpoint'ов: /api/learning/level-test/*, /api/learning/puzzle/check."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.app import app

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


class TestLevelTestStart:
    def test_without_puzzle_base(self):
        with patch(MATCH_ANCHOR, None):
            resp = client.post("/api/learning/level-test/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База паззлов не загружена"
        assert data.get("questions") == []

    def test_returns_questions_without_solutions(self):
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post("/api/learning/level-test/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert all("fen" in q and "id" in q and "themes" in q for q in data["questions"])
        assert all("moves" not in q and "solution" not in q for q in data["questions"])


class TestLevelTestCheck:
    def test_without_puzzle_base(self):
        with patch(MATCH_ANCHOR, None):
            resp = client.post(
                "/api/learning/level-test/check",
                json={"puzzle_id": "a1", "move": "b1b8"},
            )
        assert resp.status_code == 200
        assert resp.json().get("error") == "База паззлов не загружена"

    def test_correct_answer(self):
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/level-test/check",
                json={"puzzle_id": "b5", "move": "b1b8"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct"] is True
        assert data["solution"] == "b1b8"

    def test_wrong_answer(self):
        with NO_STOCKFISH:
            with patch(MATCH_ANCHOR, NEW_PUZZLES):
                resp = client.post(
                    "/api/learning/level-test/check",
                    json={"puzzle_id": "b5", "move": "b1b1"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct"] is False

    def test_unknown_puzzle(self):
        with patch(MATCH_ANCHOR, NEW_PUZZLES):
            resp = client.post(
                "/api/learning/level-test/check",
                json={"puzzle_id": "zzz", "move": "b1b8"},
            )
        assert resp.status_code == 200
        assert resp.json().get("error") == "Задача не найдена"


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
