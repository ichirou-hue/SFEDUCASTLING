"""Тесты endpoint'ов базы знаний: /api/knowledge/openings, /api/knowledge/opening, /api/knowledge/random-opening, /api/knowledge/check-move."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- /api/knowledge/openings ---

class TestGetOpenings:
    def test_without_knowledge_base(self):
        with patch("backend.api_gateway.state.knowledge_base", None):
            resp = client.get("/api/knowledge/openings")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["error"] == "База знаний не загружена"


# --- /api/knowledge/opening ---

class TestGetOpeningByFen:
    def test_without_knowledge_base(self):
        with patch("backend.api_gateway.state.knowledge_base", None):
            resp = client.get("/api/knowledge/opening", params={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена"

    def test_empty_fen(self):
        with patch("backend.api_gateway.state.knowledge_base", None):
            resp = client.get("/api/knowledge/opening", params={"fen": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена" or data.get("opening") is None


# --- /api/knowledge/random-opening ---

class TestGetRandomOpening:
    def test_without_knowledge_base(self):
        with patch("backend.api_gateway.state.knowledge_base", None):
            resp = client.get("/api/knowledge/random-opening")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена"


# --- /api/knowledge/check-move ---

class TestCheckMove:
    def test_without_knowledge_base(self):
        with patch("backend.api_gateway.state.knowledge_base", None):
            resp = client.post("/api/knowledge/check-move", json={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "База знаний не загружена"

    def test_invalid_fen(self):
        resp = client.post("/api/knowledge/check-move", json={"fen": "bad"})
        assert resp.status_code == 422


NEW_OPENINGS = {
    "openings": [
        {
            "eco": "B20", "name": "Sicilian Deep", "pgn": "1. e4 c5 2. Nf3 d6 3. d4",
            "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],
            "fens": [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR",
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R",
                "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R",
                "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R",
            ],
            "fen": "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R",
        },
        {
            "eco": "B20", "name": "Sicilian Shallow", "pgn": "1. e4 c5",
            "moves": ["e2e4", "c7c5"],
            "fens": [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
                "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR",
            ],
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR",
        },
    ]
}

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
AFTER_D4 = "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4"
MATCH_ANCHOR = "backend.api_gateway.state.knowledge_base"


class TestDeepMatch:
    def test_returns_deepest_opening(self):
        with patch(MATCH_ANCHOR, NEW_OPENINGS):
            resp = client.get("/api/knowledge/opening", params={"fen": AFTER_D4})
        assert resp.status_code == 200
        assert resp.json()["opening"]["name"] == "Sicilian Deep"

    def test_shallow_position_prefers_deep_line(self):
        with patch(MATCH_ANCHOR, NEW_OPENINGS):
            resp = client.get("/api/knowledge/opening", params={"fen": AFTER_E4})
        assert resp.status_code == 200
        assert resp.json()["opening"]["name"] == "Sicilian Deep"

    def test_check_move_book_next(self):
        with patch(MATCH_ANCHOR, NEW_OPENINGS):
            resp = client.post("/api/knowledge/check-move", json={"fen": AFTER_E4})
        assert resp.status_code == 200
        data = resp.json()
        assert data["in_theory"] is True
        assert data["opening"] == "Sicilian Deep"
        assert data["next_moves"] == ["c7c5"]

    def test_check_move_midline_next(self):
        with patch(MATCH_ANCHOR, NEW_OPENINGS):
            resp = client.post("/api/knowledge/check-move", json={"fen": AFTER_D4})
        assert resp.status_code == 200
        data = resp.json()
        assert data["in_theory"] is True
        assert data["eco"] == "B20"
        assert data["next_moves"] == []

    def test_check_move_shallow_final(self):
        shallow_final = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        with patch(MATCH_ANCHOR, NEW_OPENINGS):
            resp = client.post("/api/knowledge/check-move", json={"fen": shallow_final})
        assert resp.status_code == 200
        data = resp.json()
        assert data["in_theory"] is True
        assert data["next_moves"] == ["g1f3"]
