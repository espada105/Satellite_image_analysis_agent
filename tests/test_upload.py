from fastapi.testclient import TestClient

from orchestrator_api.main import app


def test_upload_image_with_verified_user(monkeypatch) -> None:
    monkeypatch.setenv("VERIFIED_USER_IDS", "alice")

    client = TestClient(app)

    response = client.post(
        "/upload-image",
        headers={"x-user-id": "alice"},
        files={"file": ("sample.png", b"fakepng", "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["image_uri"].startswith("data/imagery/uploads/")
