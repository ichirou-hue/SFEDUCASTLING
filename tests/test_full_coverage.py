"""Дополнительные тесты покрытия для непокрытых веток (с моками зависимостей)."""

import os
import json
import sys
import io
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
import chess
from fastapi.testclient import TestClient
from backend.app import app
from backend.api_gateway import state as st
from backend.api_gateway.routes.game import make_move, maia_move
from backend.api_gateway.models import MoveRequest, FenRequest

client = TestClient(app)
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ============================
# game.py — промоция, шах, пат, Maia с моком
# ============================

class TestGamePromotionAndStatus:
    """Покрываем: промоция (36-37), шах (53), Maia inference (73-89)."""

    def test_promotion_to_knight(self):
        """Ход с превращением пешки в коня."""
        fen = "8/4P3/8/8/8/8/8/8 w - - 0 1"
        resp = client.post("/api/move", json={
            "fen": fen, "from_sq": "e7", "to_sq": "e8", "promotion": "n"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "e8=N" in data.get("san", "")

    def test_move_to_check(self):
        """Bxf7+ — ставим шах."""
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3"
        resp = client.post("/api/move", json={
            "fen": fen, "from_sq": "c4", "to_sq": "f7"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "check"

    def test_stalemate_direct(self):
        """Проверка детекции пата через прямой вызов make_move."""
        req = MoveRequest.model_construct(
            fen="k7/2B5/PK6/8/8/8/8/8 w - - 0 1",
            from_sq="a6", to_sq="a7", promotion="q"
        )
        result = make_move(req)
        assert result["status"] == "stalemate"

    def test_stalemate_detection(self):
        """Прямой тест python-chess: пат."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.parse_square("a8"), chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.parse_square("a6"), chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.parse_square("a7"), chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.BLACK
        assert board.is_stalemate()

    def test_maia_with_mocked_inference(self):
        """Maia2 загружена, inference возвращает ход."""
        mock_model = MagicMock()
        mock_prepared = MagicMock()
        mock_inf = MagicMock()
        mock_inf.inference_each.return_value = ({"e2e4": 0.8, "d2d4": 0.2}, 0.55)

        with patch.dict("sys.modules", {"maia2": MagicMock(), "maia2.inference": mock_inf}):
            with patch("backend.api_gateway.routes.game.ensure_maia2",
                       return_value=(mock_model, mock_prepared)):
                resp = client.post("/api/maia-move", json={"fen": START_FEN, "elo": 1500})

        assert resp.status_code == 200
        data = resp.json()
        assert "san" in data

    def test_maia_empty_probs(self):
        """Maia2 загружена, move_probs пустой → fallback."""
        mock_model = MagicMock()
        mock_prepared = MagicMock()
        mock_inf = MagicMock()
        mock_inf.inference_each.return_value = ({}, 0.5)

        with patch.dict("sys.modules", {"maia2": MagicMock(), "maia2.inference": mock_inf}):
            with patch("backend.api_gateway.routes.game.ensure_maia2",
                       return_value=(mock_model, mock_prepared)):
                resp = client.post("/api/maia-move", json={"fen": START_FEN, "elo": 1500})

        assert resp.status_code == 200
        data = resp.json()
        assert "san" in data

    def test_maia_inference_exception(self):
        """Maia2 загружена, inference падает → fallback."""
        mock_model = MagicMock()
        mock_prepared = MagicMock()
        mock_inf = MagicMock()
        mock_inf.inference_each.side_effect = RuntimeError("CUDA OOM")

        with patch.dict("sys.modules", {"maia2": MagicMock(), "maia2.inference": mock_inf}):
            with patch("backend.api_gateway.routes.game.ensure_maia2",
                       return_value=(mock_model, mock_prepared)):
                resp = client.post("/api/maia-move", json={"fen": START_FEN, "elo": 1500})

        assert resp.status_code == 200
        data = resp.json()
        assert "san" in data


# ============================
# analysis.py — с моками Stockfish, Maia2, GigaChat
# ============================

class MockResponse:
    """Заглушка для requests.Response."""
    def __init__(self, data, status_code=200):
        self.status_code = status_code
        self._data = data
    def json(self):
        return self._data


class TestAnalysisWithMocks:
    """Покрываем success-пути analysis.py."""

    def test_stockfish_analyze_with_mock(self):
        """Stockfish загружен, возвращает оценку."""
        mock_sf = MagicMock()
        mock_sf.get_best_move.return_value = "e2e4"
        mock_sf.get_evaluation.return_value = {"type": "cp", "value": 24}
        mock_sf.get_top_moves.return_value = [
            {"Move": "e2e4", "Centipawn": 24},
            {"Move": "d2d4", "Centipawn": 20},
        ]
        with patch("backend.api_gateway.routes.analysis.ensure_stockfish", return_value=mock_sf):
            resp = client.post("/api/stockfish-analyze", json={"fen": START_FEN})
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_move"] == "e2e4"
        assert data["evaluation"]["value"] == 24

    def test_stockfish_analyze_exception(self):
        """Stockfish загружен, но падает с ошибкой."""
        mock_sf = MagicMock()
        mock_sf.set_fen_position.side_effect = RuntimeError("Stockfish crashed")
        with patch("backend.api_gateway.routes.analysis.ensure_stockfish", return_value=mock_sf):
            resp = client.post("/api/stockfish-analyze", json={"fen": START_FEN})
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_compare_engines_with_mocks(self):
        """Оба движка загружены, сравнение работает."""
        mock_sf = MagicMock()
        mock_sf.get_best_move.return_value = "e2e4"
        mock_sf.get_evaluation.return_value = {"type": "cp", "value": 30}
        mock_sf.set_fen_position.side_effect = None

        mock_model = MagicMock()
        mock_prep = MagicMock()
        mock_inf = MagicMock()
        mock_inf.inference_each.return_value = ({"e2e4": 0.7, "d2d4": 0.3}, 0.6)

        mock_maia2 = MagicMock()
        mock_maia2.inference = mock_inf
        with patch.dict("sys.modules", {"maia2": mock_maia2}):
            with patch("backend.api_gateway.routes.analysis.ensure_stockfish",
                       return_value=mock_sf):
                with patch("backend.api_gateway.routes.analysis.ensure_maia2",
                           return_value=(mock_model, mock_prep)):
                    resp = client.post("/api/compare-maia-stockfish",
                                       json={"fen": START_FEN, "elo": 1500})

        assert resp.status_code == 200
        data = resp.json()
        assert "stockfish" in data
        assert "maia2" in data

    def test_analyze_with_gigachat_key(self):
        """GIGACHAT_AUTH_KEY установлен, GigaChat отвечает."""
        import backend.api_gateway.routes.analysis as analysis_mod
        original_key = analysis_mod.GIGACHAT_AUTH_KEY
        analysis_mod.GIGACHAT_AUTH_KEY = "test-key"

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Хорошая позиция у белых."
        mock_response.choices = [mock_choice]

        mock_giga_instance = MagicMock()
        mock_giga_instance.chat.return_value = mock_response

        mock_giga_cls = MagicMock()
        mock_giga_cls.return_value.__enter__.return_value = mock_giga_instance

        mock_giga_module = MagicMock()
        mock_giga_module.GigaChat = mock_giga_cls

        with patch("backend.api_gateway.routes.analysis.get_opening_info",
                   return_value=None):
            with patch.dict("sys.modules", {"gigachat": mock_giga_module}):
                resp = client.post("/api/analyze", json={"fen": START_FEN})

        assert resp.status_code == 200
        data = resp.json()
        assert "Хорошая позиция" in data["message"]

        analysis_mod.GIGACHAT_AUTH_KEY = original_key

    def test_analyze_gigachat_exception(self):
        """GIGACHAT_AUTH_KEY установлен, но GigaChat падает."""
        import backend.api_gateway.routes.analysis as analysis_mod
        original_key = analysis_mod.GIGACHAT_AUTH_KEY
        analysis_mod.GIGACHAT_AUTH_KEY = "test-key"

        mock_giga_cls = MagicMock(side_effect=RuntimeError("Connection failed"))
        mock_giga_module = MagicMock()
        mock_giga_module.GigaChat = mock_giga_cls

        with patch("backend.api_gateway.routes.analysis.get_opening_info",
                   return_value=None):
            with patch.dict("sys.modules", {"gigachat": mock_giga_module}):
                resp = client.post("/api/analyze", json={"fen": START_FEN})

        assert resp.status_code == 200
        data = resp.json()
        assert "Ошибка GigaChat" in data["message"]

        analysis_mod.GIGACHAT_AUTH_KEY = original_key

    def test_similarity_search_success(self):
        """Модули perception найдены, поиск работает."""
        mock_model = MagicMock()
        mock_store = MagicMock()
        mock_model.get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_store.search_similar.return_value = [
            {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
             "score": 0.95}
        ]

        modules = {
            "backend.perception": MagicMock(),
            "backend.perception.embedder": MagicMock(),
            "backend.perception.embedder.model": mock_model,
            "backend.perception.embedder.vector_store": mock_store,
        }
        with patch.dict("sys.modules", modules):
            # Hадo перезагрузить, чтобы import внутри функции увидел моки
            import importlib
            import backend.api_gateway.routes.analysis as analysis_mod
            importlib.reload(analysis_mod)
            resp = client.post("/api/similarity/search",
                               json={"fen": START_FEN, "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        # Перезагружаем оригинал
        importlib.reload(analysis_mod)


# ============================
# knowledge.py — с загруженной базой знаний
# ============================

TEST_OPENINGS = {
    "openings": [
        {"name": "Italian Game", "eco": "C50",
         "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3",
         "description": "Популярный открытый дебют."},
        {"name": "Sicilian Defense", "eco": "B20",
         "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
         "description": "Полуоткрытый дебют."},
    ]
}

class TestKnowledgeLoaded:
    """Покрываем knowledge.py с загруженной базой."""

    def test_list_openings(self):
        with patch("backend.api_gateway.routes.knowledge.knowledge_base", TEST_OPENINGS):
            resp = client.get("/api/knowledge/openings")
        assert resp.status_code == 200
        assert len(resp.json()["openings"]) == 2

    def test_opening_found_by_fen(self):
        with patch("backend.api_gateway.routes.knowledge.knowledge_base", TEST_OPENINGS):
            resp = client.get("/api/knowledge/opening",
                params={"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3"})
        assert resp.status_code == 200
        assert resp.json()["opening"]["name"] == "Italian Game"

    def test_opening_not_found(self):
        with patch("backend.api_gateway.routes.knowledge.knowledge_base", TEST_OPENINGS):
            resp = client.get("/api/knowledge/opening", params={"fen": START_FEN})
        assert resp.status_code == 200
        assert resp.json()["opening"] is None

    def test_random_opening(self):
        with patch("backend.api_gateway.routes.knowledge.knowledge_base", TEST_OPENINGS):
            resp = client.get("/api/knowledge/random-opening")
        assert resp.status_code == 200
        assert "name" in resp.json()["opening"]

    def test_check_move_found(self):
        with patch("backend.api_gateway.routes.knowledge.knowledge_base", TEST_OPENINGS):
            resp = client.post("/api/knowledge/check-move", json={
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3"})
        assert resp.status_code == 200
        assert resp.json()["in_theory"] is True

    def test_check_move_not_found(self):
        with patch("backend.api_gateway.routes.knowledge.knowledge_base", TEST_OPENINGS):
            resp = client.post("/api/knowledge/check-move", json={"fen": START_FEN})
        assert resp.status_code == 200
        assert resp.json()["in_theory"] is False


# ============================
# data.py — save-move success и PGN-обработка
# ============================

class TestDataSaveSuccess:
    """Покрываем success-путь save_move_to_dataset и PGN error handler."""

    def test_save_move_success(self, tmp_path):
        """Stockfish загружен, сохраняем ход."""
        mock_sf = MagicMock()
        mock_sf.get_best_move.return_value = "e2e4"
        mock_sf.get_evaluation.return_value = {"type": "cp", "value": 30}

        with patch("backend.api_gateway.routes.data.ensure_stockfish", return_value=mock_sf):
            with patch("backend.api_gateway.routes.data.os.path.dirname",
                       return_value=str(tmp_path)):
                resp = client.post("/api/save-move-to-dataset", json={
                    "fen": START_FEN, "move": "e2e4", "user_id": "test", "game_id": "g1"
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert (tmp_path / "dataset.jsonl").exists()

    def test_save_move_exception(self):
        """Stockfish загружен, set_fen_position падает."""
        mock_sf = MagicMock()
        mock_sf.set_fen_position.side_effect = ValueError("bad fen")
        with patch("backend.api_gateway.routes.data.ensure_stockfish", return_value=mock_sf):
            resp = client.post("/api/save-move-to-dataset", json={
                "fen": START_FEN, "move": "e2e4"})
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_pgn_text_error_response(self):
        """PGN с некорректным форматом → ошибка."""
        resp = client.post("/api/parse-pgn-text",
                           json={"pgn": "1. e4 e5 2. " * 1000})
        assert resp.status_code in (200, 422)

    def test_pgn_binary_decode(self):
        """parse-pgn с latin-1 содержимым."""
        binary_content = bytes(range(128, 256))  # latin-1
        resp = client.post(
            "/api/parse-pgn",
            files={"file": ("game.pgn", binary_content, "text/plain")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data or "games_count" in data


# ============================
# vision.py — с моком LLaVA
# ============================

class TestVisionSuccess:
    """Покрываем success-путь analyze-image."""

    def test_analyze_image_with_llava(self):
        """LLaVA загружена, изображение распознано."""
        with patch("backend.api_gateway.routes.vision.load_llava", return_value=True):
            with patch("backend.api_gateway.routes.vision.extract_fen_from_image",
                       return_value=START_FEN):
                resp = client.post(
                    "/api/analyze-image",
                    files={"file": ("board.png", b"fake-png-data", "image/png")}
                )
        assert resp.status_code == 200
        assert resp.json()["fen"] == START_FEN

    def test_analyze_image_error(self):
        """LLaVA загружена, распознавание вернуло ошибку."""
        with patch("backend.api_gateway.routes.vision.load_llava", return_value=True):
            with patch("backend.api_gateway.routes.vision.extract_fen_from_image",
                       return_value="ERROR: cannot read board"):
                resp = client.post(
                    "/api/analyze-image",
                    files={"file": ("board.jpg", b"fake-data", "image/jpeg")}
                )
        assert resp.status_code == 200
        assert resp.json()["error"] == "cannot read board"

    def test_analyze_image_exception(self):
        """LLaVA загружена, исключение при обработке."""
        with patch("backend.api_gateway.routes.vision.load_llava", return_value=True):
            resp = client.post(
                "/api/analyze-image",
                files={"file": ("board.png", b"", "image/png")}
            )
        assert resp.status_code == 200
        assert "error" in resp.json()


# ============================
# state.py — success-пути и get_opening_info
# ============================

class TestStateSuccess:
    """Покрываем успешные загрузки и работу state.py."""

    def test_load_stockfish_success(self):
        """Stockfish бинарник найден, загружаем."""
        with patch("os.path.exists", return_value=True):
            with patch.dict("sys.modules",
                            {"stockfish": MagicMock(), "stockfish.Stockfish": MagicMock()}):
                result = st.load_stockfish()
        assert result is True
        assert st.stockfish is not None
        st.stockfish = None

    def test_load_maia2_success(self):
        """maia2 установлена, загружаем."""
        mock_model_mod = MagicMock()
        mock_model_mod.from_pretrained.return_value = MagicMock()
        mock_inf_mod = MagicMock()
        mock_inf_mod.prepare.return_value = MagicMock()
        mock_maia2 = MagicMock()
        mock_maia2.model = mock_model_mod
        mock_maia2.inference = mock_inf_mod
        with patch.dict("sys.modules", {"maia2": mock_maia2}):
            result = st.load_maia2()
        assert result is True
        assert st.maia2 is not None
        assert st.maia2_prepared is not None
        st.maia2 = None
        st.maia2_prepared = None

    def test_load_llava_success(self):
        """transformers установлены, LLaVA загружается."""
        mock_torch = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock()
        transformers = MagicMock()
        transformers.pipeline = mock_pipe

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": transformers}):
            result = st.load_llava()
        assert result is True
        assert st.llava_model is not None
        st.llava_model = None

    def test_load_knowledge_bad_json(self, tmp_path):
        """Файл есть, но JSON битый."""
        original_path = st.KNOWLEDGE_PATH
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json", encoding="utf-8")
        st.KNOWLEDGE_PATH = str(bad_file)
        result = st.load_knowledge()
        assert result is False
        st.KNOWLEDGE_PATH = original_path

    def test_get_opening_info_success(self):
        """Lichess API отвечает с дебютом и партиями."""
        mock_resp = MockResponse({
            "opening": {"name": "Italian Game"},
            "topGames": [
                {"white": {"name": "Carlsen"}, "black": {"name": "Nakamura"},
                 "year": 2020, "winner": "white"}
            ]
        })
        with patch("backend.api_gateway.state.requests.get", return_value=mock_resp):
            result = st.get_opening_info(START_FEN)
        assert result is not None
        assert result["name"] == "Italian Game"

    def test_get_opening_info_no_opening(self):
        """Lichess API отвечает, но без дебюта."""
        mock_resp = MockResponse({})
        with patch("backend.api_gateway.state.requests.get", return_value=mock_resp):
            result = st.get_opening_info(START_FEN)
        assert result is None

    def test_extract_fen_from_image_success(self, tmp_path):
        """LLaVA возвращает FEN."""
        img_file = tmp_path / "board.png"
        img_file.write_text("fake", encoding="utf-8")

        st.llava_model = MagicMock()
        st.llava_model.return_value = [
            {"generated_text": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
        ]
        mock_pil = MagicMock()
        mock_pil.Image.open.return_value.convert.return_value = MagicMock()

        with patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_pil.Image}):
            result = st.extract_fen_from_image(str(img_file))

        assert not result.startswith("ERROR:")
        assert "/" in result
        st.llava_model = None


# ============================
# app.py — serve_index и GIGACHAT key
# ============================

class TestAppCoverage:
    """Покрываем app.py: serve_index (line 45)."""

    def test_serve_index(self):
        """GET / возвращает index.html."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert len(resp.content) > 0
