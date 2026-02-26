from fastapi import Header, HTTPException, status

from packages.shared.app_shared.common.config import get_verified_user_ids


def require_verified_user(x_user_id: str | None = Header(default=None)) -> str | None:
    allowed_ids = get_verified_user_ids()
    if not allowed_ids:
        return x_user_id
    if not x_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="x-user-id header is required")
    if x_user_id not in allowed_ids:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user is not verified")
    return x_user_id
