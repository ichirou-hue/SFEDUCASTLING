"""Тесты модуля state.py: ModelManager, загрузка моделей, fallback-поведение."""

import os
import json
from unittest.mock import patch, MagicMock
import pytest

from backend.api_gateway import state


@pytest.fixture(autouse=True)
def reset_manager():
    """Сбрасывает ModelManager перед каждым тестом."""
    state.manager._stockfish = None
    state.manager._stockfish_loaded = False


# --- ModelManager: Stockfish ---

class TestModelManagerStockfish:
    def test_get_stockfish_without_binary(self):
        with patch.object(state.manager, "_load_stockfish"):
            result = state.manager.get_stockfish()
        assert result is None

    def test_get_stockfish_twice(self):
        mock = MagicMock()
        # Новый state.py проверяет живость процесса через
        # engine._stockfish.poll(): None = процесс жив.
        mock._stockfish.poll.return_value = None
        state.manager._stockfish = mock

        first = state.manager.get_stockfish()
        second = state.manager.get_stockfish()

        assert first is mock
        assert second is mock

    def test_get_stockfish_load_called_once(self):
        def side_effect():
            engine = MagicMock()
            # Процесс «жив», чтобы не сработал авто-рестарт.
            engine._stockfish.poll.return_value = None
            state.manager._stockfish = engine

        with patch.object(state.manager, "_load_stockfish",
                          side_effect=side_effect) as mock_load:
            state.manager.get_stockfish()
            state.manager.get_stockfish()

        mock_load.assert_called_once()

    def test_ensure_stockfish_not_loaded(self):
        with patch("backend.api_gateway.state.manager._load_stockfish"):
            result = state.ensure_stockfish()
            assert result is None

    def test_ensure_stockfish_already_loaded(self):
        mock = MagicMock()
        # Процесс «жив» — ensure должен вернуть тот же экземпляр.
        mock._stockfish.poll.return_value = None
        state.manager._stockfish = mock

        result = state.ensure_stockfish()

        assert result is mock


class TestBackwardCompatLoadStockfish:
    def test_load_without_binary(self):
        with patch.object(state.manager, "_load_stockfish"):
            result = state.load_stockfish()
        assert result is False

    def test_load_twice(self):
        mock = MagicMock()
        mock._stockfish.poll.return_value = None
        state.manager._stockfish = mock

        result = state.load_stockfish()

        assert result is True


# --- Загрузка LLaVA ---

class TestLoadLlava:
    def test_load_without_transformers(self):
        result = state.load_llava()
        assert result is False
        assert state.llava_model is None

    def test_load_twice(self):
        result = state.load_llava()
        assert result is False
        assert state.llava_model is None


# --- Загрузка базы знаний ---

class TestLoadKnowledge:
    def test_load_without_file(self):
        result = state.load_knowledge()
        assert result is False
        assert state.knowledge_base is None

    def test_load_with_file(self, tmp_path):
        openings = {"openings": [{"name": "Italian Game", "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"}]}
        knowledge_dir = os.path.join(os.path.dirname(state.KNOWLEDGE_PATH), "..", "knowledge")
        os.makedirs(knowledge_dir, exist_ok=True)
        original_path = state.KNOWLEDGE_PATH
        test_path = tmp_path / "openings.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(openings, f)

        state.KNOWLEDGE_PATH = str(test_path)
        result = state.load_knowledge()
        assert result is True
        assert state.knowledge_base is not None
        assert len(state.knowledge_base["openings"]) == 1
        state.knowledge_base = None
        state.KNOWLEDGE_PATH = original_path


# --- extract_fen_from_image ---

class TestExtractFenFromImage:
    def test_without_llava(self):
        state.llava_model = None
        with patch("backend.api_gateway.state.load_llava", return_value=False):
            result = state.extract_fen_from_image("fake.jpg")
            assert result.startswith("ERROR:")

    def test_file_not_found(self):
        state.llava_model = MagicMock()
        with patch("backend.api_gateway.state.load_llava", return_value=True):
            result = state.extract_fen_from_image("nonexistent.png")
            assert "ERROR" in result.upper()

