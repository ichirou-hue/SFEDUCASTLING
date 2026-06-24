"""Тесты endpoint'ов компьютерного зрения: /api/analyze-image."""

import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


# --- /api/analyze-image ---

class TestAnalyzeImage:
    def test_without_llava(self):
        resp = client.post(
            "/api/analyze-image",
            files={"file": ("board.png", b"fake-image-data", "image/png")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert "LLaVA не загружена" in data["error"]

    def test_no_file(self):
        resp = client.post("/api/analyze-image")
        assert resp.status_code == 422

    def test_wrong_extension(self):
        resp = client.post(
            "/api/analyze-image",
            files={"file": ("board.txt", b"data", "text/plain")}
        )
        # Годятся только PNG/JPG/BMP/WEBP — txt будет отклонён моделью на этапе LLaVA
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
