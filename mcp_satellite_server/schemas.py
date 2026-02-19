from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_uri: str
    ops: list[str] = Field(default_factory=lambda: ["edges"])
    roi: dict | None = None


class OpResult(BaseModel):
    name: str
    summary: str
    stats: dict[str, float] = Field(default_factory=dict)
    artifact_uri: str | None = None


class AnalyzeResponse(BaseModel):
    ops: list[OpResult] = Field(default_factory=list)
