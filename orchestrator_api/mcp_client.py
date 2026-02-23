import json
import uuid
from urllib.parse import urljoin

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
    direct_error: str | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            try:
                # Prefer MCP streamable HTTP (/mcp). This is the FastMCP standard path.
                rpc = await _call_via_direct_rpc(client, call_payload)
            except Exception as exc:  # noqa: BLE001
                direct_error = str(exc)
                rpc = await _call_via_sse_transport(client, call_payload)
    except Exception as exc:  # noqa: BLE001
        if direct_error:
            return AnalysisResult(invoked=True, error=f"direct={direct_error}; sse={exc}")
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


async def _call_via_sse_transport(client: httpx.AsyncClient, payload: dict) -> dict:
    sse_url = f"{settings.mcp_base_url}/mcp/sse"
    async with client.stream("GET", sse_url) as stream_response:
        stream_response.raise_for_status()
        line_iter = stream_response.aiter_lines()

        endpoint_event = await _read_sse_event(line_iter)
        if endpoint_event.get("event") != "endpoint":
            raise RuntimeError("invalid sse bootstrap event")

        endpoint_data = endpoint_event.get("data") or {}
        message_url = endpoint_data.get("message_url")
        if not message_url:
            raise RuntimeError("missing message_url from sse endpoint event")

        post_url = urljoin(f"{settings.mcp_base_url.rstrip('/')}/", str(message_url).lstrip("/"))
        post_resp = await client.post(post_url, json=payload)
        post_resp.raise_for_status()

        rpc_event = await _read_sse_event(line_iter)
        if rpc_event.get("event") != "message":
            raise RuntimeError("expected message event from sse transport")
        return rpc_event.get("data") or {}


async def _read_sse_event(line_iter) -> dict:
    event_name = "message"
    event_data: dict | None = None

    async for line in line_iter:
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            raw_data = line[len("data: ") :]
            event_data = _safe_json_load(raw_data)
        elif line == "":
            if event_data is None:
                event_name = "message"
                continue
            return {"event": event_name, "data": event_data}

    raise RuntimeError("sse stream ended before event complete")


def _safe_json_load(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to parse sse json payload: {exc}") from exc
    if isinstance(value, dict):
        return value
    raise RuntimeError("sse payload is not a json object")
