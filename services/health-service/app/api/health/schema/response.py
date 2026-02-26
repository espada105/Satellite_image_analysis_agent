from pydantic import BaseModel


class ServiceHealthResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, str]
