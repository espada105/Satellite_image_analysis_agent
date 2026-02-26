from pydantic import BaseModel


class GatewayHealthResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, str]
