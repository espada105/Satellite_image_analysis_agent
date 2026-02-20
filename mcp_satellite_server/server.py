from fastapi import FastAPI

from mcp_satellite_server.opencv_ops import SUPPORTED_OPS, analyze_satellite_image
from mcp_satellite_server.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    McpRpcRequest,
    McpRpcResponse,
)

app = FastAPI(title="Satellite MCP Server", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mcp", response_model=McpRpcResponse)
def mcp_rpc(payload: McpRpcRequest) -> McpRpcResponse:
    if payload.method == "initialize":
        return McpRpcResponse(
            id=payload.id,
            result={
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "satellite-mcp", "version": "0.2.0"},
                "capabilities": {"tools": {}},
            },
        )

    if payload.method == "tools/list":
        return McpRpcResponse(
            id=payload.id,
            result={
                "tools": [
                    {
                        "name": "analyze_satellite_image",
                        "description": "Analyze satellite image via OpenCV heuristics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "image_uri": {"type": "string"},
                                "ops": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": sorted(SUPPORTED_OPS)},
                                },
                                "roi": {"type": "object"},
                            },
                            "required": ["image_uri", "ops"],
                        },
                    }
                ]
            },
        )

    if payload.method == "tools/call":
        name = payload.params.get("name")
        arguments = payload.params.get("arguments", {})
        if name != "analyze_satellite_image":
            return McpRpcResponse(
                id=payload.id,
                error={"code": -32602, "message": f"Unknown tool: {name}"},
            )

        req = AnalyzeRequest.model_validate(arguments)
        ops = analyze_satellite_image(req.image_uri, req.ops, req.roi)
        result = AnalyzeResponse(ops=ops).model_dump()
        return McpRpcResponse(id=payload.id, result={"content": [{"type": "json", "json": result}]})

    return McpRpcResponse(
        id=payload.id,
        error={"code": -32601, "message": f"Method not found: {payload.method}"},
    )


# Backward-compatible REST endpoint
@app.post("/tools/analyze_satellite_image", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    results = analyze_satellite_image(payload.image_uri, payload.ops, payload.roi)
    return AnalyzeResponse(ops=results)
