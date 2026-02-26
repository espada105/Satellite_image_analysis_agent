from fastapi import APIRouter

from app.service.health.triton import check_triton

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    triton = await check_triton()
    status = "ok" if triton == "ok" else "degraded"
    return {"status": status, "service": "health-service", "dependencies": {"triton": triton}}
