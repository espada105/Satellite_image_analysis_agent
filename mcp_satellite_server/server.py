from fastapi import FastAPI

from mcp_satellite_server.opencv_ops import analyze_satellite_image
from mcp_satellite_server.schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="Satellite MCP Server", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tools/analyze_satellite_image", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    results = analyze_satellite_image(payload.image_uri, payload.ops, payload.roi)
    return AnalyzeResponse(ops=results)
