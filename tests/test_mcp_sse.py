from fastapi.testclient import TestClient

from mcp_satellite_server.server import app


def test_mcp_sse_unknown_session_returns_404() -> None:
    client = TestClient(app)
    resp = client.post(
        "/mcp/messages/not_found_session",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert resp.status_code == 404
