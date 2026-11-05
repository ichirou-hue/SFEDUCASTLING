from fastapi.testclient import TestClient

from backend.api_gateway.routes import health
from backend.app import app


def test_health_endpoint_confirms_database_readiness(monkeypatch):
    monkeypatch.setattr(health, "stockfish_ready", lambda: True)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_rejects_missing_stockfish(monkeypatch):
    monkeypatch.setattr(health, "stockfish_ready", lambda: False)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Stockfish unavailable"
