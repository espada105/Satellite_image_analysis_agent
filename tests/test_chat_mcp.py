from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from mcp_satellite_server.server import app as mcp_app
from orchestrator_api.main import app as orchestrator_app
from orchestrator_api.rag.store import store


def test_chat_with_mcp_flow(tmp_path: Path, monkeypatch) -> None:
    headers = {"x-user-id": "alice"}
    store.clear()

    doc_path = tmp_path / "manual.txt"
    doc_path.write_text("Change detection manual", encoding="utf-8")

    image_path = tmp_path / "image.png"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:70, 20:70] = (255, 255, 255)
    cv2.imwrite(str(image_path), image)

    mcp_client = TestClient(mcp_app)

    async def _fake_analyze_image(
        image_uri: str,
        ops: list[str],
        roi=None,
        timeout_s: float = 20.0,
    ):
        response = mcp_client.post(
            "/tools/analyze_satellite_image",
            json={"image_uri": image_uri, "ops": ops, "roi": roi},
        )
        data = response.json()
        from orchestrator_api.schemas import AnalysisOpSummary, AnalysisResult

        return AnalysisResult(
            invoked=True,
            ops=[
                AnalysisOpSummary(
                    name=item["name"],
                    summary=item["summary"],
                    stats=item.get("stats", {}),
                )
                for item in data["ops"]
            ],
        )

    monkeypatch.setattr("orchestrator_api.services.chat_service.analyze_image", _fake_analyze_image)

    client = TestClient(orchestrator_app)
    client.post(
        "/ingest",
        json={"documents": [str(doc_path)]},
        headers=headers,
    )

    chat_resp = client.post(
        "/chat",
        json={
            "question": "이 지역 변화 탐지해줘",
            "image_uri": str(image_path),
            "ops": ["edges"],
        },
        headers=headers,
    )

    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["analysis"]["invoked"] is True
    assert body["analysis"]["ops"][0]["name"] == "edges"
