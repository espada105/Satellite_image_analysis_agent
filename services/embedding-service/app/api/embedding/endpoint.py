from fastapi import APIRouter, Depends

from app.api.embedding.schema.request import EmbeddingRequest
from app.api.embedding.schema.response import EmbeddingResponse
from app.security.auth import require_verified_user
from app.service.embedding import generate_embeddings
from packages.shared.app_shared.schema.health import HealthResponse

router = APIRouter(tags=["embedding"])


@router.post("/embed", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest, _: str | None = Depends(require_verified_user)) -> EmbeddingResponse:
    return generate_embeddings(request)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="embedding-service")
