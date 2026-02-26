from fastapi import FastAPI

from app.api.upload.endpoint import router as upload_router

app = FastAPI(title="File Service", version="0.1.0")
app.include_router(upload_router)
