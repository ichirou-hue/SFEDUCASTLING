from io import BytesIO
import uuid

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
import backend.services.avatar_service as avatar_service

client = TestClient(app)


def _register_user() -> tuple[str, dict]:
    login = f"avatar_{uuid.uuid4().hex[:10]}"
    response = client.post(
        "/api/auth/register",
        json={"login": login, "password": "Sup3rSecret!"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["access_token"], body["user"]


def _png_bytes(size=(128, 128)) -> bytes:
    out = BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(out, format="PNG")
    return out.getvalue()


def _jpeg_bytes(size=(128, 128)) -> bytes:
    out = BytesIO()
    Image.new("RGB", size, (20, 90, 140)).save(out, format="JPEG")
    return out.getvalue()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_avatar_upload_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    response = client.post(
        "/api/avatars/me",
        files={"file": ("avatar.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 401


def test_valid_png_is_reencoded_to_server_webp(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()

    original = _png_bytes()
    response = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.png", original, "image/png")},
    )
    assert response.status_code == 200, response.text

    user = response.json()["user"]
    assert user["avatar_url"].startswith("/api/avatars/")
    assert len(list(tmp_path.glob("*.webp"))) == 1

    fetched = client.get(user["avatar_url"])
    assert fetched.status_code == 200
    assert fetched.headers["content-type"].startswith("image/webp")
    assert fetched.headers["x-content-type-options"] == "nosniff"
    assert fetched.content[:4] == b"RIFF"
    assert fetched.content[8:12] == b"WEBP"
    assert fetched.content != original


def test_fake_png_payload_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()
    response = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.png", b"<?php echo 'x'; ?>", "image/png")},
    )
    assert response.status_code == 415
    assert list(tmp_path.iterdir()) == []


def test_extension_content_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()
    response = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.png", _jpeg_bytes(), "image/png")},
    )
    assert response.status_code == 415


def test_svg_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    response = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.svg", svg, "image/svg+xml")},
    )
    assert response.status_code == 415


def test_oversized_upload_is_rejected_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()
    payload = b"\xff\xd8\xff" + b"x" * avatar_service.MAX_UPLOAD_BYTES
    response = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413


def test_excessive_dimensions_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()
    response = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.png", _png_bytes((4097, 1)), "image/png")},
    )
    assert response.status_code == 422


def test_delete_removes_avatar_and_invalidates_old_url(tmp_path, monkeypatch):
    monkeypatch.setattr(avatar_service, "AVATAR_DIR", tmp_path)
    token, _ = _register_user()

    uploaded = client.post(
        "/api/avatars/me",
        headers=_auth(token),
        files={"file": ("avatar.png", _png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 200
    old_url = uploaded.json()["user"]["avatar_url"]

    deleted = client.delete("/api/avatars/me", headers=_auth(token))
    assert deleted.status_code == 200
    assert deleted.json()["user"]["avatar_url"] is None
    assert client.get(old_url).status_code == 404
    assert list(tmp_path.glob("*.webp")) == []
