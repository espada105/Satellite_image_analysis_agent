import asyncio
import json
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from mcp_satellite_server.opencv_ops import SUPPORTED_OPS, analyze_satellite_image
from mcp_satellite_server.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    McpRpcRequest,
    McpRpcResponse,
)

app = FastAPI(title="Satellite MCP Server", version="0.3.0")
_SESSIONS: dict[str, asyncio.Queue[str]] = {}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\\ndata: {json.dumps(data, ensure_ascii=False)}\\n\\n"


def _dispatch_rpc(payload: McpRpcRequest) -> McpRpcResponse:
    if payload.method == "initialize":
        return McpRpcResponse(
            id=payload.id,
            result={
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "satellite-mcp", "version": "0.3.0"},
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mcp", response_model=McpRpcResponse)
def mcp_rpc(payload: McpRpcRequest) -> McpRpcResponse:
    return _dispatch_rpc(payload)


@app.get("/mcp/sse")
async def mcp_sse() -> StreamingResponse:
    session_id = uuid4().hex
    queue: asyncio.Queue[str] = asyncio.Queue()
    _SESSIONS[session_id] = queue

    async def event_generator():
        try:
            yield _sse(
                "endpoint",
                {
                    "session_id": session_id,
                    "message_url": f"/mcp/messages/{session_id}",
                },
            )
            while True:
                event_text = await queue.get()
                yield event_text
        finally:
            _SESSIONS.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/mcp/messages/{session_id}")
async def mcp_messages(session_id: str, payload: McpRpcRequest) -> dict[str, str]:
    queue = _SESSIONS.get(session_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Unknown MCP session")

    response = _dispatch_rpc(payload).model_dump()
    await queue.put(_sse("message", response))
    return {"status": "accepted"}


# Backward-compatible REST endpoint
@app.post("/tools/analyze_satellite_image", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    results = analyze_satellite_image(payload.image_uri, payload.ops, payload.roi)
    return AnalyzeResponse(ops=results)
