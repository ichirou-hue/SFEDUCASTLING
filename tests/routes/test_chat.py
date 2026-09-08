"""Тесты чата: маркер доски [ДОСКА: FEN] и endpoint /api/chat/ask."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app import app
from backend.api_gateway.routes.chat import extract_board_fen, _mentions_position

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK1NR b KQkq - 3 3"


# --- extract_board_fen ---

class TestExtractBoardFen:
    def test_no_marker(self):
        text, fen = extract_board_fen("Просто текстовый ответ без доски.")
        assert text == "Просто текстовый ответ без доски."
        assert fen is None

    def test_valid_marker(self):
        raw = f"Итальянская партия.\n[ДОСКА: {ITALIAN_FEN}]"
        text, fen = extract_board_fen(raw)
        assert fen == ITALIAN_FEN
        assert "ДОСКА" not in text
        assert text == "Итальянская партия."

    def test_invalid_marker_kept_in_text(self):
        raw = "Что-то там\n[ДОСКА: это не fen]"
        text, fen = extract_board_fen(raw)
        assert fen is None
        assert "ДОСКА" in text

    def test_marker_in_middle(self):
        raw = f"Сначала позиция [ДОСКА: {START_FEN}] и продолжение."
        text, fen = extract_board_fen(raw)
        assert fen == START_FEN
        assert "Сначала позиция" in text
        assert "и продолжение." in text

    def test_only_marker(self):
        text, fen = extract_board_fen(f"[ДОСКА: {ITALIAN_FEN}]")
        assert fen == ITALIAN_FEN
        assert text == ""


# --- _mentions_position ---

class TestMentionsPosition:
    def test_position_keywords(self):
        assert _mentions_position("Какой лучший ход сейчас?")
        assert _mentions_position("Что делать в этой позиции?")
        assert _mentions_position("Куда ходить ферзём?")
        assert _mentions_position("Оцени мат в этой позиции")

    def test_non_position_keywords(self):
        assert not _mentions_position("Привет! Как дела?")
        assert not _mentions_position("Какая сегодня погода?")
        assert not _mentions_position("Спасибо за помощь!")
        assert not _mentions_position("Расскажи что-нибудь про музыку")

    def test_empty_message_is_non_position(self):
        assert not _mentions_position("")


# --- /api/chat/ask ---

class TestChatAsk:
    def test_ask_with_board_marker(self):
        engine = MagicMock()
        engine.chat.return_value = f"Конечно, вот позиция.\n[ДОСКА: {ITALIAN_FEN}]"
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine):
            resp = client.post("/api/chat/ask", json={
                "message": "Покажи позицию после итальянской.",
                "fen": START_FEN,
                "is_greeting": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Конечно, вот позиция."
        assert data["fen"] == ITALIAN_FEN

    def test_ask_without_marker(self):
        engine = MagicMock()
        engine.chat.return_value = "Простой текстовый ответ."
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine):
            resp = client.post("/api/chat/ask", json={"message": "Как ходит конь?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Простой текстовый ответ."
        assert data["fen"] is None

    def test_ask_attaches_current_position_when_no_marker(self):
        engine = MagicMock()
        engine.chat.return_value = "Совет по позиции: развивай фигуры."
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine):
            resp = client.post("/api/chat/ask", json={
                "message": "Что делать в этой позиции?",
                "fen": START_FEN,
                "is_greeting": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN

    def test_ask_non_chess_question_attaches_no_board(self):
        engine = MagicMock()
        engine.chat.return_value = "Сегодня хорошая погода."
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine):
            resp = client.post("/api/chat/ask", json={
                "message": "Какая сегодня погода?",
                "fen": START_FEN,
                "is_greeting": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Сегодня хорошая погода."
        assert data["fen"] is None

    def test_ask_position_question_attaches_board(self):
        engine = MagicMock()
        engine.chat.return_value = "Развивайте коня."
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine):
            resp = client.post("/api/chat/ask", json={
                "message": "Что делать в этой позиции?",
                "fen": START_FEN,
                "is_greeting": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN

    def test_no_engine(self):
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=None):
            resp = client.post("/api/chat/ask", json={"message": "Вопрос"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] is None
        assert "не подключён" in data["reply"]

    def test_invalid_fen_in_marker_does_not_crash(self):
        engine = MagicMock()
        engine.chat.return_value = "Позиция не распознана [ДОСКА: abc123]."
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine):
            resp = client.post("/api/chat/ask", json={"message": "Хм"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] is None
        assert "abc123" in data["reply"]

    def test_ask_returns_deterministic_stockfish_arrow(self):
        engine = MagicMock()
        engine.chat.return_value = "Развивай фигуры в центре."
        fake_sf = MagicMock()
        fake_sf.get_best_move.return_value = "e2e4"
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine), \
                patch("backend.api_gateway.routes.chat.ensure_stockfish", return_value=fake_sf), \
                patch("backend.api_gateway.routes.chat.stockfish_lock"):
            resp = client.post("/api/chat/ask", json={
                "message": "Какой лучший ход?",
                "fen": START_FEN,
                "is_greeting": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] == START_FEN
        assert data["arrow"] == {
            "uci": "e2e4",
            "san": "e4",
            "from": "e2",
            "to": "e4",
        }

    def test_ask_greeting_has_no_arrow(self):
        engine = MagicMock()
        engine.chat.return_value = "Привет!"
        fake_sf = MagicMock()
        fake_sf.get_best_move.return_value = "e2e4"
        with patch("backend.api_gateway.routes.chat.get_gigachess", return_value=engine), \
                patch("backend.api_gateway.routes.chat.ensure_stockfish", return_value=fake_sf):
            resp = client.post("/api/chat/ask", json={
                "message": "Привет",
                "fen": START_FEN,
                "is_greeting": True,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["fen"] is None
        assert data["arrow"] is None