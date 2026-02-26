from fastapi import FastAPI

from app.api.health.endpoint import router as health_router

app = FastAPI(title="Health Service", version="0.1.0")
app.include_router(health_router)
