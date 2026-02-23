import json
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

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    with TestClient(app) as client:
        init_resp = client.post(
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
        assert init_resp.status_code == 200
        server_info = init_resp.json()["result"].get("serverInfo", {})
        assert server_info.get("name")
        init_notify_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            headers=headers,
        )
        assert init_notify_resp.status_code in (200, 202, 204)

        list_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=headers,
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
            headers=headers,
        )
        assert call_resp.status_code == 200
        result = call_resp.json().get("result", {})
        data = result.get("structuredContent")
        if not isinstance(data, dict):
            content = result.get("content") or []
            assert content
            data = content[0].get("json", {})
            if not data:
                data = json.loads(content[0].get("text", "{}"))
        assert data["ops"][0]["name"] == "edges"
