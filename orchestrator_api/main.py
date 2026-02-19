from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from orchestrator_api.config import settings
from orchestrator_api.rag.ingest import ingest_documents
from orchestrator_api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from orchestrator_api.security import require_verified_user
from orchestrator_api.services.chat_service import run_chat

app = FastAPI(title="Satellite Orchestrator API", version="0.1.0")
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mcp_base_url": settings.mcp_base_url}


@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/chatbot", include_in_schema=False)
def chatbot_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "chatbot.html")


@app.get("/auth/verify")
def auth_verify(_: str | None = Depends(require_verified_user)) -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    _: str | None = Depends(require_verified_user),
) -> IngestResponse:
    ingested_count, failed = ingest_documents(
        request.documents,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    return IngestResponse(ingested_count=ingested_count, failed=failed)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: str | None = Depends(require_verified_user),
) -> ChatResponse:
    return await run_chat(request)
