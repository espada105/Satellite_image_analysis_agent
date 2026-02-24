from types import SimpleNamespace

import pytest

from orchestrator_api.llm import decide_image_ops, decide_tool_usage, generate_answer_with_llm
from orchestrator_api.schemas import AnalysisResult


@pytest.mark.asyncio
async def test_decide_tool_usage_falls_back_when_langchain_and_legacy_fail(monkeypatch):
    monkeypatch.setattr(
        "orchestrator_api.llm.settings",
        SimpleNamespace(llm_api_key="x", llm_model="gpt-4o-mini"),
    )
    monkeypatch.setattr("orchestrator_api.llm.langchain_enabled", lambda: True)

    class _FailingChain:
        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("lc down")

    monkeypatch.setattr(
        "orchestrator_api.llm._router_chain",
        lambda timeout_s=12.0: _FailingChain(),
    )

    async def _fail_legacy(*_args, **_kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("orchestrator_api.llm._legacy_json_prompt_call", _fail_legacy)

    decision = await decide_tool_usage("질문", image_available=True)
    assert decision.use_rag is True
    assert decision.use_mcp is True
    assert decision.reason == "rule_fallback"
    assert decision.error is not None


@pytest.mark.asyncio
async def test_generate_answer_returns_error_not_raise_when_both_paths_fail(monkeypatch):
    monkeypatch.setattr(
        "orchestrator_api.llm.settings",
        SimpleNamespace(llm_api_key="x", llm_model="gpt-4o-mini"),
    )
    monkeypatch.setattr("orchestrator_api.llm.langchain_enabled", lambda: True)

    class _FailingChain:
        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("lc down")

    monkeypatch.setattr(
        "orchestrator_api.llm._answer_chain",
        lambda timeout_s=20.0: _FailingChain(),
    )

    async def _fail_legacy(*_args, **_kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("orchestrator_api.llm._legacy_text_prompt_call", _fail_legacy)

    answer, err = await generate_answer_with_llm("q", [], AnalysisResult(invoked=False))
    assert answer is None
    assert err is not None


@pytest.mark.asyncio
async def test_decide_image_ops_falls_back_when_langchain_and_legacy_fail(monkeypatch):
    monkeypatch.setattr(
        "orchestrator_api.llm.settings",
        SimpleNamespace(llm_api_key="x", llm_model="gpt-4o-mini"),
    )
    monkeypatch.setattr("orchestrator_api.llm.langchain_enabled", lambda: True)

    class _FailingChain:
        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("lc down")

    monkeypatch.setattr(
        "orchestrator_api.llm._ops_selector_chain",
        lambda timeout_s=12.0: _FailingChain(),
    )

    async def _fail_legacy(*_args, **_kwargs):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr("orchestrator_api.llm._legacy_json_prompt_call", _fail_legacy)

    ops, reason = await decide_image_ops("경계 분석")
    assert ops
    assert reason.startswith("ops_rule_fallback:")
