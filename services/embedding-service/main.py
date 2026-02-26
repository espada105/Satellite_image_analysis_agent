from fastapi import FastAPI

from app.api.embedding.endpoint import router as embedding_router

app = FastAPI(title="Embedding Service", version="0.1.0")
app.include_router(embedding_router)
