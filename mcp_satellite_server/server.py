from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastmcp import FastMCP

from mcp_satellite_server.opencv_ops import analyze_satellite_image
from mcp_satellite_server.schemas import AnalyzeRequest, AnalyzeResponse, McpRpcRequest

mcp = FastMCP(name="satellite-mcp", version="0.4.0")
mcp_app = mcp.http_app(
    path="/mcp",
    json_response=True,
    stateless_http=True,
    transport="http",
)


@mcp.tool(
    name="analyze_satellite_image",
    description="Analyze satellite image via OpenCV heuristics",
)
def analyze_satellite_image_tool(
    image_uri: str,
    ops: list[str],
    roi: dict | None = None,
) -> dict:
    req = AnalyzeRequest(image_uri=image_uri, ops=ops, roi=roi)
    results = analyze_satellite_image(req.image_uri, req.ops, req.roi)
    return AnalyzeResponse(ops=results).model_dump()


app = FastAPI(title="Satellite MCP Server", version="0.4.0", lifespan=mcp_app.lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Backward-compatible REST endpoint for existing orchestrator/tests.
@app.post("/tools/analyze_satellite_image", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    results = analyze_satellite_image(payload.image_uri, payload.ops, payload.roi)
    return AnalyzeResponse(ops=results)


# Legacy custom SSE transport endpoint is retired in FastMCP mode.
@app.get("/mcp/sse")
def mcp_sse_deprecated() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={"detail": "legacy /mcp/sse transport removed; use streamable HTTP on /mcp"},
    )


# Legacy endpoint kept for explicit compatibility with old tests/clients.
@app.post("/mcp/messages/{session_id}")
async def mcp_messages_deprecated(session_id: str, payload: McpRpcRequest) -> dict[str, str]:
    del payload
    raise HTTPException(status_code=404, detail=f"Unknown MCP session: {session_id}")


app.mount("/", mcp_app)
