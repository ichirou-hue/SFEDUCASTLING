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


# --- /api/eval ---

class TestQuickEval:
    BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    WHITE_TO_MOVE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"

    def test_without_stockfish(self):
        with patch("backend.api_gateway.routes.analysis.ensure_stockfish", return_value=None):
            resp = client.post("/api/eval", json={"fen": START_FEN})
        assert resp.status_code == 200
        assert resp.json() == {"evaluation": None}

    def test_invalid_fen(self):
        resp = client.post("/api/eval", json={"fen": "bad"})
        assert resp.status_code == 422

    def test_sign_is_always_from_white(self):
        """Движок отвечает от лица ходящей стороны, роут - от лица белых."""

        class Engine:
            def set_fen_position(self, fen):
                self.fen = fen

            def get_evaluation(self, searchtime=None):
                # Ходящая сторона видит перевес у себя: и при ходе
                # чёрных, и при ходе белых движок скажет "+300".
                return {"type": "cp", "value": 300}

        with patch(
            "backend.api_gateway.routes.analysis.ensure_stockfish",
            return_value=Engine(),
        ):
            black = client.post("/api/eval", json={"fen": self.BLACK_TO_MOVE}).json()
            white = client.post("/api/eval", json={"fen": self.WHITE_TO_MOVE}).json()

        assert white["evaluation"]["value"] == 300
        assert black["evaluation"]["value"] == -300

    def test_mate_sign_flips_too(self):
        class Engine:
            def set_fen_position(self, fen):
                self.fen = fen

            def get_evaluation(self, searchtime=None):
                return {"type": "mate", "value": 3}

        with patch(
            "backend.api_gateway.routes.analysis.ensure_stockfish",
            return_value=Engine(),
        ):
            black = client.post("/api/eval", json={"fen": self.BLACK_TO_MOVE}).json()

        assert black["evaluation"] == {"type": "mate", "value": -3}


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
