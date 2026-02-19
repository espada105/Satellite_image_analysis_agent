from pathlib import Path

from fastapi.testclient import TestClient

from orchestrator_api.main import app
from orchestrator_api.rag.store import store


def test_chat_rag_only_flow(tmp_path: Path) -> None:
    store.clear()
    doc_path = tmp_path / "faq.txt"
    doc_path.write_text("Cloud detection and boundary extraction basics", encoding="utf-8")

    client = TestClient(app)

    ingest_resp = client.post("/ingest", json={"documents": [str(doc_path)]})
    assert ingest_resp.status_code == 200

    chat_resp = client.post("/chat", json={"question": "cloud detection 설명해줘", "top_k": 2})
    assert chat_resp.status_code == 200

    body = chat_resp.json()
    assert "answer" in body
    assert len(body["citations"]) >= 1
    assert body["analysis"]["invoked"] is False
