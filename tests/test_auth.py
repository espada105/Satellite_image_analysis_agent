from pathlib import Path

from fastapi.testclient import TestClient

from orchestrator_api.main import app
from orchestrator_api.rag.store import store


def test_verified_user_only_access(tmp_path: Path, monkeypatch) -> None:
    store.clear()
    doc_path = tmp_path / "auth_doc.txt"
    doc_path.write_text("cloud detection notes", encoding="utf-8")

    monkeypatch.setenv("VERIFIED_USER_IDS", "alice,bob")

    client = TestClient(app)

    no_header_resp = client.post("/ingest", json={"documents": [str(doc_path)]})
    assert no_header_resp.status_code == 401

    bad_user_resp = client.post(
        "/ingest",
        json={"documents": [str(doc_path)]},
        headers={"x-user-id": "mallory"},
    )
    assert bad_user_resp.status_code == 403

    ok_resp = client.post(
        "/ingest",
        json={"documents": [str(doc_path)]},
        headers={"x-user-id": "alice"},
    )
    assert ok_resp.status_code == 200
