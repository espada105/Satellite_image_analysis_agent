from fastapi import FastAPI

from app.api.retrieve.endpoint import router as retrieve_router

app = FastAPI(title="Retrieve Service", version="0.1.0")
app.include_router(retrieve_router)
