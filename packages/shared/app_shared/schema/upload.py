from pydantic import BaseModel


class UploadRelayRequest(BaseModel):
    filename: str
    content_type: str
    content_base64: str


class UploadResponse(BaseModel):
    image_uri: str
    preview_url: str
