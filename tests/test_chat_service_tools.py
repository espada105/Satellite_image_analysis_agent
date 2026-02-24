from orchestrator_api.schemas import AnalysisOpSummary, AnalysisResult
from orchestrator_api.services.chat_service import _append_mcp_tools_to_answer


def test_append_mcp_tools_when_invoked() -> None:
    analysis = AnalysisResult(
        invoked=True,
        ops=[AnalysisOpSummary(name="edges", summary="ok"), 
        AnalysisOpSummary(name="threshold", summary="ok")],
    )
    out = _append_mcp_tools_to_answer("answer", analysis)
    assert "MCP tools: edges, threshold" in out


def test_append_mcp_tools_failed() -> None:
    analysis = AnalysisResult(invoked=True, error="timeout")
    out = _append_mcp_tools_to_answer("answer", analysis)
    assert "MCP tools: (failed)" in out
