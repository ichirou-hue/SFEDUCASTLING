"""Тесты игровых endpoint'ов: /api/legal-moves, /api/move, /api/stockfish-move."""

import chess
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

    def test_valid_request(self, monkeypatch):
        monkeypatch.setattr(
            "backend.api_gateway.routes.game._stockfish_move",
            lambda board: chess.Move.from_uci("e2e4"),
        )
        resp = client.post("/api/stockfish-move", json={"fen": START_FEN, "elo": 1500, "engine": "stockfish"})
        assert resp.status_code == 200
        data = resp.json()
        assert "fen" in data
        assert "san" in data
        assert "from" in data
        assert "to" in data
        assert data.get("engine") == "stockfish"

    def test_unavailable_maia_does_not_fall_back_to_stockfish(self, monkeypatch):
        monkeypatch.setattr(
            "backend.api_gateway.routes.game._maia3_move",
            lambda req, board: None,
        )
        monkeypatch.setattr(
            "backend.api_gateway.routes.game._stockfish_move",
            lambda board: pytest.fail("Maia3 must not be replaced by Stockfish"),
        )

        resp = client.post(
            "/api/maia-move",
            json={"fen": START_FEN, "elo": 1500, "engine": "maia3"},
        )

        assert resp.status_code == 503

    def test_levels_list(self):
        resp = client.get("/api/levels")
        assert resp.status_code == 200
        data = resp.json()
        assert "levels" in data
        assert len(data["levels"]) >= 7
        assert data["levels"][0]["engine"] == "maia3"
        assert data["levels"][-1]["engine"] == "stockfish"

    def test_invalid_fen(self):
        resp = client.post("/api/stockfish-move", json={"fen": "bad", "elo": 1500})
        assert resp.status_code == 422


# --- /api/game/new ---

class TestNewGame:
    def test_default_is_white(self):
        resp = client.post("/api/game/new", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN
        assert data["player_color"] == "w"
        assert data["ai_color"] == "b"
        assert data["turn"] == "w"
        assert data["ai_to_move"] is False

    def test_playing_black_ai_moves_first(self):
        resp = client.post("/api/game/new", json={"player_color": "b"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["player_color"] == "b"
        assert data["ai_color"] == "w"
        assert data["ai_to_move"] is True

    def test_accepts_human_readable_color(self):
        resp = client.post("/api/game/new", json={"player_color": "черные"})
        assert resp.status_code == 200
        assert resp.json()["player_color"] == "b"

    def test_invalid_color(self):
        resp = client.post("/api/game/new", json={"player_color": "green"})
        assert resp.status_code == 422


# --- /api/game/captured-material ---

class TestCapturedMaterial:
    def test_empty_history(self):
        resp = client.post("/api/game/captured-material", json={"moves": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN
        assert data["captured_by_white"] == []
        assert data["captured_by_black"] == []
        assert data["material"]["diff"] == 0
        assert data["material"]["advantage"] == "equal"

    def test_normal_capture(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e4", "d7d5", "e4d5"], "player_color": "w"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured_by_white"] == ["p"]
        assert data["captured_by_black"] == []
        assert data["counts"]["white"]["p"] == 1
        assert data["material"]["diff"] == 1
        assert data["material"]["advantage"] == "white"
        assert data["player_advantage"] == 1

    def test_player_advantage_for_black(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e4", "d7d5", "e4d5"], "player_color": "b"},
        )
        assert resp.status_code == 200
        assert resp.json()["player_advantage"] == -1

    def test_en_passant_capture(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured_by_white"] == ["p"]
        assert data["counts"]["white"]["p"] == 1

    def test_promotion_capture_is_not_a_capture_of_own_piece(self):
        # 1.d4 h5 2.c4 h4 3.Qd3 h3 4.Qxh3 — ферзь берёт пешку.
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["d2d4", "h7h5", "c2c4", "h5h4", "d1d3", "h4h3", "d3h3"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured_by_white"] == ["p"]
        assert data["material"]["white"] == 1

    def test_illegal_move_returns_400(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": ["e2e5"]},
        )
        assert resp.status_code == 400

    def test_invalid_fen(self):
        resp = client.post(
            "/api/game/captured-material",
            json={"moves": [], "fen": "bad"},
        )
        assert resp.status_code == 422


# --- /api/game/finish ---

class TestFinishGame:
    def test_accepts_player_color(self):
        # Пустой список ходов: фоновый record_game выходит без обращения к БД.
        resp = client.post(
            "/api/game/finish",
            json={"moves": [], "player_color": "b"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_invalid_player_color(self):
        resp = client.post(
            "/api/game/finish",
            json={"moves": [], "player_color": "green"},
        )
        assert resp.status_code == 422

