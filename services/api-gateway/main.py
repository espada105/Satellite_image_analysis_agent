from fastapi import FastAPI

from app.api.health.endpoint import router as health_router
from app.api.retrieve.endpoint import router as retrieve_router
from app.api.upload.endpoint import router as upload_router
from app.security.cors import setup_cors

app = FastAPI(title="API Gateway", version="0.1.0")
setup_cors(app)
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(retrieve_router)
