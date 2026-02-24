import json
import uuid

import httpx

from orchestrator_api.config import settings
from orchestrator_api.schemas import AnalysisOpSummary, AnalysisResult


async def analyze_image(
    image_uri: str,
    ops: list[str],
    roi: dict | None = None,
    timeout_s: float = 20.0,
) -> AnalysisResult:
    call_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": "analyze_satellite_image",
            "arguments": {"image_uri": image_uri, "ops": ops, "roi": roi},
        },
    }

    rpc: dict | None = None
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            rpc = await _call_via_direct_rpc(client, call_payload)
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(invoked=True, error=str(exc))

    if not rpc:
        return AnalysisResult(invoked=True, error="empty rpc response")

    if rpc.get("error"):
        return AnalysisResult(invoked=True, error=rpc["error"].get("message", "mcp error"))

    data = _extract_tool_data(rpc)
    if not data:
        return AnalysisResult(invoked=True, error="empty mcp result")

    ops_out: list[AnalysisOpSummary] = []
    for item in data.get("ops", []):
        ops_out.append(
            AnalysisOpSummary(
                name=item.get("name", "unknown"),
                summary=item.get("summary", ""),
                artifact_uri=item.get("artifact_uri"),
                stats=item.get("stats", {}),
            )
        )
    return AnalysisResult(invoked=True, ops=ops_out)


def _extract_tool_data(rpc: dict) -> dict:
    result = rpc.get("result") or {}

    # FastMCP json_response=True returns structured JSON here.
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    # Legacy custom server and some MCP servers encode JSON content in result.content.
    content = result.get("content") or []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                json_blob = item.get("json")
                if isinstance(json_blob, dict):
                    return json_blob
                text_blob = item.get("text")
                if isinstance(text_blob, str):
                    try:
                        parsed = json.loads(text_blob)
                    except Exception:  # noqa: BLE001
                        continue
                    if isinstance(parsed, dict):
                        return parsed
    return {}


async def _call_via_direct_rpc(client: httpx.AsyncClient, payload: dict) -> dict:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "orchestrator", "version": "0.1.0"},
        },
    }
    await client.post(f"{settings.mcp_base_url}/mcp", json=init_payload, headers=headers)
    await client.post(
        f"{settings.mcp_base_url}/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        headers=headers,
    )
    response = await client.post(f"{settings.mcp_base_url}/mcp", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()
