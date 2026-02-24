import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from mcp_satellite_server.opencv_ops import ARTIFACT_DIR
from mcp_satellite_server.server import create_app


def test_mcp_analyze_satellite_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[10:60, 10:60] = (255, 255, 255)
    cv2.imwrite(str(image_path), image)

    with TestClient(create_app()) as client:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-client", "version": "0.1.0"},
                },
            },
            headers=headers,
        )
        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            headers=headers,
        )
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "analyze_satellite_image",
                    "arguments": {
                        "image_uri": str(image_path),
                        "ops": ["edges", "cloud_mask_like", "masking_like"],
                    },
                },
            },
            headers=headers,
        )

    assert response.status_code == 200
    result = response.json().get("result", {})
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        content = result.get("content") or []
        assert content
        data = content[0].get("json", {})
        if not data:
            data = json.loads(content[0].get("text", "{}"))
    assert len(data["ops"]) == 3
    assert data["ops"][0]["name"] == "edges"
    assert data["ops"][2]["name"] == "masking_like"
    for op in data["ops"]:
        assert op["artifact_uri"] is not None
        assert op["artifact_uri"].startswith("/imagery/artifacts/")
        artifact_name = op["artifact_uri"].split("/imagery/artifacts/", 1)[1]
        artifact_path = ARTIFACT_DIR / artifact_name
        assert artifact_path.exists()
