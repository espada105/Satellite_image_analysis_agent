from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from orchestrator_api.llm import ToolDecision
from orchestrator_api.main import app
from orchestrator_api.rag.store import store
from orchestrator_api.schemas import AnalysisOpSummary, AnalysisResult


def test_chat_uses_langchain_pipeline_path(tmp_path: Path, monkeypatch) -> None:
    headers = {"x-user-id": "alice"}
    store.clear()
    doc_path = tmp_path / "faq.txt"
    doc_path.write_text("Cloud detection and boundary extraction basics", encoding="utf-8")

    async def _fake_decide_tool_usage(
        question: str,
        image_available: bool,
        timeout_s: float = 12.0,
    ):
        _ = (question, image_available, timeout_s)
        return ToolDecision(use_rag=True, use_mcp=False, reason="test")

    async def _fake_generate_answer(question: str, citations, analysis, timeout_s: float = 20.0):
        _ = (question, citations, analysis, timeout_s)
        return "pipeline-answer", None

    monkeypatch.setattr(
        "orchestrator_api.services.chat_service.settings",
        SimpleNamespace(use_langchain_pipeline=True),
    )
    monkeypatch.setattr(
        "orchestrator_api.services.chat_langchain_pipeline.decide_tool_usage",
        _fake_decide_tool_usage,
    )
    monkeypatch.setattr(
        "orchestrator_api.services.chat_langchain_pipeline.generate_answer_with_llm",
        _fake_generate_answer,
    )

    client = TestClient(app)
    ingest_resp = client.post("/ingest", json={"documents": [str(doc_path)]}, headers=headers)
    assert ingest_resp.status_code == 200

    chat_resp = client.post(
        "/chat",
        json={"question": "cloud detection 설명해줘", "top_k": 2},
        headers=headers,
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["answer"].startswith("pipeline-answer")
    assert "route.reason:test" in body["trace"]["tools"]


def test_chat_pipeline_uses_mcp_structured_tool(monkeypatch) -> None:
    headers = {"x-user-id": "alice"}

    class _FakeTool:
        async def ainvoke(self, payload: dict):
            _ = payload
            return AnalysisResult(
                invoked=True,
                ops=[AnalysisOpSummary(name="edges", summary="ok")],
            )

    async def _fake_decide_tool_usage(
        question: str,
        image_available: bool,
        timeout_s: float = 12.0,
    ):
        _ = (question, image_available, timeout_s)
        return ToolDecision(use_rag=False, use_mcp=True, reason="test")

    async def _fake_generate_answer(question: str, citations, analysis, timeout_s: float = 20.0):
        _ = (question, citations, analysis, timeout_s)
        return "mcp-pipeline-answer", None

    monkeypatch.setattr(
        "orchestrator_api.services.chat_service.settings",
        SimpleNamespace(use_langchain_pipeline=True),
    )
    monkeypatch.setattr(
        "orchestrator_api.services.chat_langchain_pipeline.build_analyze_satellite_image_tool",
        lambda: _FakeTool(),
    )
    monkeypatch.setattr(
        "orchestrator_api.services.chat_langchain_pipeline.decide_tool_usage",
        _fake_decide_tool_usage,
    )
    monkeypatch.setattr(
        "orchestrator_api.services.chat_langchain_pipeline.generate_answer_with_llm",
        _fake_generate_answer,
    )

    client = TestClient(app)
    chat_resp = client.post(
        "/chat",
        json={"question": "분석", "image_uri": "data/imagery/test_1.png", "ops": ["edges"]},
        headers=headers,
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["analysis"]["invoked"] is True
    assert body["analysis"]["ops"][0]["name"] == "edges"
