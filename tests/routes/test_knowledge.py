"""Тесты endpoint'ов базы знаний: /api/knowledge/openings, /api/knowledge/opening, /api/knowledge/random-opening, /api/knowledge/check-move."""

import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- /api/knowledge/openings ---

class TestGetOpenings:
    def test_without_knowledge_base(self):
        resp = client.get("/api/knowledge/openings")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["error"] == "База знаний не загружена"


# --- /api/knowledge/opening ---

class TestGetOpeningByFen:
    def test_without_knowledge_base(self):
        resp = client.get("/api/knowledge/opening", params={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена"

    def test_empty_fen(self):
        resp = client.get("/api/knowledge/opening", params={"fen": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена" or data.get("opening") is None


# --- /api/knowledge/random-opening ---

class TestGetRandomOpening:
    def test_without_knowledge_base(self):
        resp = client.get("/api/knowledge/random-opening")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена"


# --- /api/knowledge/check-move ---

class TestCheckMove:
    def test_without_knowledge_base(self):
        resp = client.post("/api/knowledge/check-move", json={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена"

    def test_invalid_fen(self):
        resp = client.post("/api/knowledge/check-move", json={"fen": "bad"})
        assert resp.status_code == 422
