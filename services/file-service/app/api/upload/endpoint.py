from fastapi import APIRouter, Depends

from app.api.upload.schema.request import UploadRelayRequest
from app.api.upload.schema.response import UploadResponse
from app.security.auth import require_verified_user
from app.service.upload.file import upload_encoded_image
from packages.shared.app_shared.schema.health import HealthResponse

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
def upload(request: UploadRelayRequest, _: str | None = Depends(require_verified_user)) -> UploadResponse:
    return UploadResponse(**upload_encoded_image(**request.model_dump()))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="file-service")
