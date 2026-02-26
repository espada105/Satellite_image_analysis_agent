from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]] = Field(default_factory=list)
    dimension: int = 0
