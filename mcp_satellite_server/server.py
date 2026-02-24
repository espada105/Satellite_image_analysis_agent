import os

from fastmcp import FastMCP

from mcp_satellite_server.opencv_ops import analyze_satellite_image
from mcp_satellite_server.schemas import AnalyzeRequest, AnalyzeResponse

mcp = FastMCP(name="satellite-mcp", version="0.4.0")


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


def create_app():
    return mcp.http_app(
        path="/mcp",
        json_response=True,
        stateless_http=True,
        transport="http",
    )


app = create_app()


def main() -> None:
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8100"))
    mcp.run(
        transport="http",
        host=host,
        port=port,
        path="/mcp",
        json_response=True,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
