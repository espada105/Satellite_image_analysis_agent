import httpx

from orchestrator_api.config import settings
from orchestrator_api.schemas import AnalysisOpSummary, AnalysisResult


async def analyze_image(
    image_uri: str,
    ops: list[str],
    roi: dict | None = None,
    timeout_s: float = 20.0,
) -> AnalysisResult:
    payload = {"image_uri": image_uri, "ops": ops, "roi": roi}
    url = f"{settings.mcp_base_url}/tools/analyze_satellite_image"

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        return AnalysisResult(invoked=True, error=str(exc))

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
