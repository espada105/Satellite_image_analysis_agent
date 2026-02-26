from fastapi import APIRouter, Depends

from app.common.config import settings
from app.security.auth import require_verified_user
from app.service.proxy import forward_post
from packages.shared.app_shared.schema.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("", response_model=ChatResponse)
async def retrieve(
    request: ChatRequest,
    user_id: str | None = Depends(require_verified_user),
) -> ChatResponse:
    data = await forward_post(
        f"{settings.retrieve_service_base_url}/retrieve",
        request.model_dump(),
        {"x-user-id": user_id or ""},
    )
    return ChatResponse(**data)
