"""Тесты игровых endpoint'ов: /api/legal-moves, /api/move, /api/stockfish-move."""

import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- /api/legal-moves ---

class TestLegalMoves:
    def test_valid_start_position(self):
        resp = client.post("/api/legal-moves", json={"fen": START_FEN, "square": "e2"})
        assert resp.status_code == 200
        data = resp.json()
        assert "moves" in data
        assert len(data["moves"]) == 2  # e3, e4
        assert "e3" in data["moves"]
        assert "e4" in data["moves"]

    def test_knight_moves(self):
        resp = client.post("/api/legal-moves", json={"fen": START_FEN, "square": "g1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "f3" in data["moves"]
        assert "h3" in data["moves"]

    def test_invalid_fen(self):
        resp = client.post("/api/legal-moves", json={"fen": "invalid", "square": "e2"})
        assert resp.status_code == 422

    def test_invalid_square(self):
        resp = client.post("/api/legal-moves", json={"fen": START_FEN, "square": "z9"})
        assert resp.status_code == 422


# --- /api/move ---

class TestMakeMove:
    def test_valid_pawn_push(self):
        resp = client.post("/api/move", json={
            "fen": START_FEN, "from_sq": "e2", "to_sq": "e4"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["san"] == "e4"
        assert data["status"] == "playing"
        assert data["turn"] == "b"

    def test_illegal_move(self):
        resp = client.post("/api/move", json={
            "fen": START_FEN, "from_sq": "e2", "to_sq": "e6"
        })
        # Модель MoveRequest отклоняет нелегальный ход → 422
        assert resp.status_code == 422

    def test_checkmate(self):
        # Scholar's mate: после Qh5, Nc6, Bc4, Nf6 — Qxf7#
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        resp = client.post("/api/move", json={
            "fen": fen, "from_sq": "h5", "to_sq": "f7"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "checkmate"

    def test_invalid_fen(self):
        resp = client.post("/api/move", json={
            "fen": "bad", "from_sq": "e2", "to_sq": "e4"
        })
        assert resp.status_code == 422

    def test_invalid_square(self):
        resp = client.post("/api/move", json={
            "fen": START_FEN, "from_sq": "z9", "to_sq": "e4"
        })
        assert resp.status_code == 422


# --- /api/stockfish-move ---

class TestAiMove:
    def test_game_over(self):
        # Scholar's mate — чёрные получили мат, игра окончена
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        resp = client.post("/api/stockfish-move", json={"fen": fen, "elo": 1500})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "Партия окончена"

    def test_valid_request(self):
        resp = client.post("/api/stockfish-move", json={"fen": START_FEN, "elo": 1500})
        assert resp.status_code == 200
        data = resp.json()
        assert "fen" in data
        assert "san" in data
        assert "from" in data
        assert "to" in data

    def test_invalid_fen(self):
        resp = client.post("/api/stockfish-move", json={"fen": "bad", "elo": 1500})
        assert resp.status_code == 422
