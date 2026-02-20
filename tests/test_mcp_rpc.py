from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from mcp_satellite_server.server import app


def test_mcp_jsonrpc_tools_call(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:30, 10:30] = (255, 255, 255)
    cv2.imwrite(str(image_path), image)

    client = TestClient(app)

    init_resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert init_resp.status_code == 200
    assert init_resp.json()["result"]["serverInfo"]["name"] == "satellite-mcp"

    list_resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert list_resp.status_code == 200
    tools = list_resp.json()["result"]["tools"]
    assert any(tool["name"] == "analyze_satellite_image" for tool in tools)

    call_resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "analyze_satellite_image",
                "arguments": {"image_uri": str(image_path), "ops": ["edges"]},
            },
        },
    )
    assert call_resp.status_code == 200
    content = call_resp.json()["result"]["content"]
    assert content[0]["json"]["ops"][0]["name"] == "edges"
