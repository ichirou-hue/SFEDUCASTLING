"""Тесты endpoint'ов данных: /api/save-move-to-dataset, /api/parse-pgn-text, /api/parse-pgn."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app import app
from backend.config.settings import settings

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

SAMPLE_PGN = (
    '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6\n'
    '4. Ba4 Nf6 5. O-O Be7 6. Re1 b5\n'
    '7. Bb3 d6 8. c3 O-O'
)


# --- /api/save-move-to-dataset ---

class TestSaveMoveToDataset:
    def test_requires_auth(self):
        resp = client.post("/api/save-move-to-dataset", json={
            "fen": START_FEN, "move": "e2e4"
        })
        assert resp.status_code == 401

    def test_without_stockfish(self, auth_headers):
        with patch("backend.api_gateway.routes.data.ensure_stockfish", return_value=None):
            resp = client.post("/api/save-move-to-dataset", json={
                "fen": START_FEN, "move": "e2e4", "user_id": 1, "game_id": "g1"
            }, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "Stockfish не загружен"

    def test_invalid_fen(self, auth_headers):
        resp = client.post("/api/save-move-to-dataset", json={
            "fen": "bad", "move": "e2e4"
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_empty_move(self, auth_headers):
        resp = client.post("/api/save-move-to-dataset", json={
            "fen": START_FEN, "move": ""
        }, headers=auth_headers)
        assert resp.status_code == 422

    def test_user_id_from_body_is_ignored(self, auth_headers):
        """Даже если в теле передан чужой user_id, запись идёт от текущего."""
        fake = MagicMock()
        fake.get_best_move_time.return_value = "e2e4"
        fake.get_evaluation.return_value = {"type": "cp", "value": 10}
        with patch("backend.api_gateway.routes.data.ensure_stockfish", return_value=fake), \
                patch("backend.api_gateway.routes.data.stockfish_lock"):
            resp = client.post("/api/save-move-to-dataset", json={
                "fen": START_FEN, "move": "e2e4", "user_id": 999999
            }, headers=auth_headers)
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["user_id"] != 999999

    def test_default_ids(self, auth_headers):
        resp = client.post("/api/save-move-to-dataset", json={
            "fen": START_FEN, "move": "e2e4"
        }, headers=auth_headers)
        assert resp.status_code == 200


class TestDatasetExport:
    def test_requires_auth(self):
        resp = client.get("/api/dataset/export")
        assert resp.status_code == 401

    def test_requires_admin(self, auth_headers):
        resp = client.get("/api/dataset/export", headers=auth_headers)
        assert resp.status_code == 403


# --- /api/parse-pgn-text ---

class TestParsePgnText:
    def test_valid_pgn(self):
        resp = client.post("/api/parse-pgn-text", json={"pgn": SAMPLE_PGN})
        assert resp.status_code == 200
        data = resp.json()
        assert data["games_count"] == 1
        game = data["games"][0]
        assert "white" in game
        assert "black" in game
        assert len(game["moves"]) > 0
        assert game["moves"][0]["move"] == "start"
        assert game["moves"][-1]["move"] == "end"

    def test_garbage_pgn(self):
        resp = client.post("/api/parse-pgn-text", json={"pgn": "this is not pgn"})
        assert resp.status_code == 200
        data = resp.json()
        # python-chess парсит любой текст как партию без ходов (имена ?)
        assert data["games_count"] == 1
        assert data["games"][0]["white"] == "?"

    def test_empty_pgn(self):
        resp = client.post("/api/parse-pgn-text", json={"pgn": ""})
        assert resp.status_code == 422

    def test_whitespace_only(self):
        resp = client.post("/api/parse-pgn-text", json={"pgn": "   "})
        assert resp.status_code == 422


# --- /api/parse-pgn ---

class TestParsePgn:
    def test_valid_pgn_file(self):
        resp = client.post(
            "/api/parse-pgn",
            files={"file": ("game.pgn", SAMPLE_PGN, "text/plain")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["games_count"] >= 1
        assert len(data["games"]) > 0
        assert data["games"][0]["moves"][0]["move"] == "start"

    def test_empty_file(self):
        resp = client.post(
            "/api/parse-pgn",
            files={"file": ("empty.pgn", "", "text/plain")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_no_file(self):
        resp = client.post("/api/parse-pgn")
        assert resp.status_code == 422

    def test_upload_too_large(self, monkeypatch):
        monkeypatch.setattr(settings.security, "max_upload_bytes", 16)
        resp = client.post(
            "/api/parse-pgn",
            files={"file": ("big.pgn", "x" * 64, "text/plain")},
        )
        assert resp.status_code == 413
