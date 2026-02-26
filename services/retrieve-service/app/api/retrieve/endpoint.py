from fastapi import APIRouter, Depends

from app.api.retrieve.schema.request import ChatRequest
from app.api.retrieve.schema.response import ChatResponse
from app.security.auth import require_verified_user
from app.service.retrieve import run_retrieve
from packages.shared.app_shared.schema.health import HealthResponse

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=ChatResponse)
def retrieve(request: ChatRequest, _: str | None = Depends(require_verified_user)) -> ChatResponse:
    return run_retrieve(request)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="retrieve-service")
