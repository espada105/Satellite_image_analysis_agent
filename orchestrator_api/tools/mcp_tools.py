from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from orchestrator_api.mcp_client import analyze_image


class AnalyzeSatelliteImageArgs(BaseModel):
    image_uri: str = Field(..., description="Image path or URI to analyze")
    ops: list[str] = Field(default_factory=list, description="Analysis ops to execute")
    roi: dict | None = Field(default=None, description="Optional region-of-interest coordinates")


async def _analyze_satellite_image_tool(
    image_uri: str,
    ops: list[str],
    roi: dict | None = None,
):
    return await analyze_image(image_uri=image_uri, ops=ops, roi=roi)


def build_analyze_satellite_image_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=_analyze_satellite_image_tool,
        name="analyze_satellite_image",
        description="Analyze satellite imagery via MCP server and return operation summaries.",
        args_schema=AnalyzeSatelliteImageArgs,
    )
