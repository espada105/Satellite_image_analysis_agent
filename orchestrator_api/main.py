import json
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from orchestrator_api.config import settings
from orchestrator_api.rag.ingest import ingest_documents
from orchestrator_api.rag.store import store
from orchestrator_api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from orchestrator_api.security import require_verified_user
from orchestrator_api.services.chat_service import run_chat, run_chat_stream

app = FastAPI(title="Satellite Orchestrator API", version="0.1.0")
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
IMAGERY_DIR = ROOT_DIR / "data" / "imagery"
DOCS_DIR = ROOT_DIR / "data" / "docs"
UPLOADS_DIR = IMAGERY_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/imagery", StaticFiles(directory=IMAGERY_DIR), name="imagery")


@app.on_event("startup")
def startup_ingest_docs() -> None:
    if store.count() > 0:
        return

    if not DOCS_DIR.exists():
        return

    docs = sorted(
        str(path)
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf", ".html", ".htm"}
    )
    if not docs:
        return

    ingest_documents(docs)


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


@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    _: str | None = Depends(require_verified_user),
) -> dict[str, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    suffix = Path(file.filename or "").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif"}:
        raise HTTPException(status_code=400, detail="Unsupported image extension")

    filename = f"{uuid4().hex}{suffix}"
    destination = UPLOADS_DIR / filename

    content = await file.read()
    destination.write_bytes(content)
    relative_uri = f"data/imagery/uploads/{filename}"

    preview_url = f"/imagery/uploads/{filename}"
    return {"image_uri": relative_uri, "preview_url": preview_url}


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


@app.post("/reindex-docs")
def reindex_docs(_: str | None = Depends(require_verified_user)) -> dict:
    if not DOCS_DIR.exists():
        return {
            "ingested_count": 0,
            "failed": [],
            "store_count": store.count(),
            "backend": store.backend_info(),
        }

    docs = sorted(
        str(path)
        for path in DOCS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf", ".html", ".htm"}
    )
    store.clear(delete_disk=True)
    ingested_count, failed = ingest_documents(docs)
    return {
        "ingested_count": ingested_count,
        "failed": failed,
        "store_count": store.count(),
        "backend": store.backend_info(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: str | None = Depends(require_verified_user),
) -> ChatResponse:
    return await run_chat(request)


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    _: str | None = Depends(require_verified_user),
) -> StreamingResponse:
    async def event_generator():
        async for event in run_chat_stream(request):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
