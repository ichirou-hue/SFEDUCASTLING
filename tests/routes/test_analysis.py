"""Тесты endpoint'ов анализа: /api/stockfish-analyze, /api/similarity/search."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- /api/stockfish-analyze ---

class TestStockfishAnalyze:
    def test_without_stockfish(self):
        with patch("backend.api_gateway.routes.analysis.ensure_stockfish", return_value=None):
            resp = client.post("/api/stockfish-analyze", json={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "Stockfish не загружен"

    def test_invalid_fen(self):
        resp = client.post("/api/stockfish-analyze", json={"fen": "bad"})
        assert resp.status_code == 422

    def test_default_elo(self):
        resp = client.post("/api/stockfish-analyze", json={"fen": START_FEN})
        assert resp.status_code == 200


# --- /api/similarity/search ---

class TestSimilaritySearch:
    def test_modules_not_available(self):
        resp = client.post("/api/similarity/search", json={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "Vector search modules not available"

    def test_invalid_fen(self):
        resp = client.post("/api/similarity/search", json={"fen": "bad"})
        assert resp.status_code == 422

    def test_custom_top_k(self):
        resp = client.post("/api/similarity/search", json={"fen": START_FEN, "top_k": 3})
        assert resp.status_code == 200

    def test_top_k_out_of_range(self):
        resp = client.post("/api/similarity/search", json={"fen": START_FEN, "top_k": 999})
        assert resp.status_code == 422
