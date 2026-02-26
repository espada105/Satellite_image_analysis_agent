from fastapi import APIRouter

from app.common.config import settings
from app.service.proxy import forward_get

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> dict:
    targets = {
        "file-service": f"{settings.file_service_base_url}/health",
        "retrieve-service": f"{settings.retrieve_service_base_url}/health",
        "embedding-service": f"{settings.embedding_service_base_url}/health",
        "health-service": f"{settings.health_service_base_url}/health",
    }
    dependencies: dict[str, str] = {}
    for name, url in targets.items():
        try:
            dependencies[name] = (await forward_get(url)).get("status", "unknown")
        except Exception:
            dependencies[name] = "unreachable"

    status = "ok" if all(v == "ok" for v in dependencies.values()) else "degraded"
    return {"status": status, "service": "api-gateway", "dependencies": dependencies}
