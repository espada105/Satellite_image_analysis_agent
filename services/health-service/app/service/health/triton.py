from app.adapter.client.http import fetch_json
from app.common.config import settings


async def check_triton() -> str:
    try:
        payload = await fetch_json(f"{settings.mcp_base_url}/health")
        return "ok" if payload.get("status") == "ok" else "degraded"
    except Exception:
        return "unreachable"
