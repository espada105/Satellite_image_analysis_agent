import base64

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.common.config import settings
from app.security.auth import require_verified_user
from app.service.proxy import forward_post
from packages.shared.app_shared.schema.upload import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    user_id: str | None = Depends(require_verified_user),
) -> UploadResponse:
    content = await file.read()
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    payload = {
        "filename": file.filename or "upload.png",
        "content_type": file.content_type,
        "content_base64": base64.b64encode(content).decode("utf-8"),
    }
    data = await forward_post(
        f"{settings.file_service_base_url}/upload",
        payload,
        {"x-user-id": user_id or ""},
    )
    return UploadResponse(**data)
