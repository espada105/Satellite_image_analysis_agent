from fastapi import FastAPI

from orchestrator_api.config import settings
from orchestrator_api.rag.ingest import ingest_documents
from orchestrator_api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from orchestrator_api.services.chat_service import run_chat

app = FastAPI(title="Satellite Orchestrator API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mcp_base_url": settings.mcp_base_url}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    ingested_count, failed = ingest_documents(
        request.documents,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    return IngestResponse(ingested_count=ingested_count, failed=failed)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await run_chat(request)
