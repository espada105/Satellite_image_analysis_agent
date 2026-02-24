from fastapi.testclient import TestClient

from mcp_satellite_server.server import create_app


def test_mcp_sse_unknown_session_returns_404() -> None:
    client = TestClient(create_app())
    resp = client.get("/mcp/sse")
    assert resp.status_code == 404
