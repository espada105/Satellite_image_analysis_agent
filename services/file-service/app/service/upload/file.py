import base64
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.adapter.repository.loader.pdf import load_bytes
from app.adapter.repository.splitter.pdf import split_payload
from app.common.config import UPLOAD_DIR

_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"}


def upload_encoded_image(filename: str, content_type: str, content_base64: str) -> dict:
    suffix = Path(filename).suffix.lower() or ".png"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported image extension")

    raw = base64.b64decode(content_base64.encode("utf-8"))
    payload = load_bytes(raw)
    split_payload(payload)

    stored_name = f"{uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / stored_name
    destination.write_bytes(payload)
    return {
        "image_uri": f"data/imagery/uploads/{stored_name}",
        "preview_url": f"/imagery/uploads/{stored_name}",
    }
