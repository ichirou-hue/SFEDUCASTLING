from backend.config.settings import ModelsSettings, Settings


def test_flat_stockfish_path_is_read(monkeypatch, tmp_path):
    executable = tmp_path / "stockfish"
    monkeypatch.setenv("STOCKFISH_PATH", str(executable))
    assert ModelsSettings(_env_file=None).stockfish_path == executable


def test_nested_stockfish_path_overrides_flat(monkeypatch, tmp_path):
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "flat"))
    nested = tmp_path / "nested"
    monkeypatch.setenv("MODELS__STOCKFISH_PATH", str(nested))
    assert Settings(_env_file=None).models.stockfish_path == nested
