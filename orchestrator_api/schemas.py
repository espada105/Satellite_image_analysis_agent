from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str
    score: float
    line_start: int | None = None
    line_end: int | None = None


class AnalysisOpSummary(BaseModel):
    name: str
    summary: str
    artifact_uri: str | None = None
    stats: dict[str, float] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    invoked: bool = False
    ops: list[AnalysisOpSummary] = Field(default_factory=list)
    error: str | None = None


class TraceInfo(BaseModel):
    tools: list[str] = Field(default_factory=list)
    latency_ms: int = 0


class ChatRequest(BaseModel):
    question: str = ""
    image_uri: str | None = None
    roi: dict | None = None
    top_k: int = 3
    ops: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    analysis: AnalysisResult = Field(default_factory=AnalysisResult)
    trace: TraceInfo = Field(default_factory=TraceInfo)


class IngestRequest(BaseModel):
    documents: list[str]
    chunk_size: int = 500
    overlap: int = 100


class IngestResponse(BaseModel):
    ingested_count: int
    failed: list[dict[str, str]] = Field(default_factory=list)
    index_name: str = "default"
