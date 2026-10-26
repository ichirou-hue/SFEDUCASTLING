from fastapi.testclient import TestClient

from backend.app import app


def test_health_endpoint_confirms_database_readiness():
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
