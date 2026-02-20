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
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            # Initialize handshake (non-fatal if it fails)
            init_payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {"clientInfo": {"name": "orchestrator", "version": "0.1.0"}},
            }
            await client.post(f"{settings.mcp_base_url}/mcp", json=init_payload)

            call_payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {
                    "name": "analyze_satellite_image",
                    "arguments": {"image_uri": image_uri, "ops": ops, "roi": roi},
                },
            }
            response = await client.post(f"{settings.mcp_base_url}/mcp", json=call_payload)
            response.raise_for_status()
            rpc = response.json()
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(invoked=True, error=str(exc))

    if rpc.get("error"):
        return AnalysisResult(invoked=True, error=rpc["error"].get("message", "mcp error"))

    content = ((rpc.get("result") or {}).get("content") or [])
    if not content:
        return AnalysisResult(invoked=True, error="empty mcp result")

    data = content[0].get("json", {})
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
