"""Тесты модуля state.py: загрузка моделей, функции-помощники, fallback-поведение."""

import os
import json
from unittest.mock import patch, MagicMock
import pytest

from backend.api_gateway import state


# --- Загрузка Stockfish ---

class TestLoadStockfish:
    def test_load_without_binary(self):
        result = state.load_stockfish()
        assert result is False
        assert state.stockfish is None

    def test_load_twice(self):
        state.stockfish = MagicMock()
        result = state.load_stockfish()
        assert result is True
        state.stockfish = None

    def test_ensure_stockfish_not_loaded(self):
        state.stockfish = None
        with patch("backend.api_gateway.state.load_stockfish", return_value=False):
            result = state.ensure_stockfish()
            assert result is None

    def test_ensure_stockfish_already_loaded(self):
        mock = MagicMock()
        state.stockfish = mock
        result = state.ensure_stockfish()
        assert result is mock
        state.stockfish = None


# --- Загрузка Maia2 ---

class TestLoadMaia2:
    def test_load_without_maia2(self):
        result = state.load_maia2()
        assert result is False
        assert state.maia2 is None
        assert state.maia2_prepared is None

    def test_load_twice(self):
        state.maia2 = MagicMock()
        state.maia2_prepared = MagicMock()
        result = state.load_maia2()
        assert result is True
        state.maia2 = None
        state.maia2_prepared = None

    def test_ensure_maia2_not_loaded(self):
        state.maia2 = None
        state.maia2_prepared = None
        with patch("backend.api_gateway.state.load_maia2", return_value=False):
            m, p = state.ensure_maia2()
            assert m is None
            assert p is None

    def test_ensure_maia2_already_loaded(self):
        mock_m = MagicMock()
        mock_p = MagicMock()
        state.maia2 = mock_m
        state.maia2_prepared = mock_p
        m, p = state.ensure_maia2()
        assert m is mock_m
        assert p is mock_p
        state.maia2 = None
        state.maia2_prepared = None


# --- Загрузка LLaVA ---

class TestLoadLlava:
    def test_load_without_transformers(self):
        result = state.load_llava()
        assert result is False
        assert state.llava_model is None

    def test_load_twice(self):
        state.llava_model = MagicMock()
        result = state.load_llava()
        assert result is True
        state.llava_model = None


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


# --- GIGACHAT_AUTH_KEY ---

class TestGigachatKey:
    def test_default_empty(self, monkeypatch):
        monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
        import importlib
        importlib.reload(state)
        assert hasattr(state, "GIGACHAT_AUTH_KEY")
        assert state.GIGACHAT_AUTH_KEY == ""
