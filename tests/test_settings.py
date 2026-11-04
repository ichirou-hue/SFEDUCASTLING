"""Загрузка вложенных настроек из переменных окружения."""

from __future__ import annotations

from pathlib import Path

from backend.config.settings import (
    DataSettings,
    GigachessSettings,
    Maia3Settings,
    ModelsSettings,
    ServerSettings,
)


def test_nested_settings_read_direct_environment_aliases(monkeypatch):
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("SERVER_PORT", "9000")
    monkeypatch.setenv("MAIA3_MODEL_ID", "maia3-test")
    monkeypatch.setenv("STOCKFISH_DEPTH", "31")
    monkeypatch.setenv("STOCKFISH_PATH", "engines/stockfish")
    monkeypatch.setenv("LLAVA_MODEL_PATH", "")
    monkeypatch.setenv("DATASET_JSONL_PATH", "custom.jsonl")
    monkeypatch.setenv("GIGACHESS_BASE_URL", "https://gigachess.example")

    assert ServerSettings().model_dump() == {"host": "0.0.0.0", "port": 9000}
    assert Maia3Settings().model_id == "maia3-test"
    models = ModelsSettings()
    assert models.stockfish_depth == 31
    assert models.stockfish_path == Path("engines/stockfish")
    assert models.llava_model_path is None
    assert DataSettings().dataset_jsonl_path == Path("custom.jsonl")
    assert GigachessSettings().base_url == "https://gigachess.example"
