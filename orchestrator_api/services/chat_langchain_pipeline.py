import time
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.runnables import RunnableLambda

from orchestrator_api.config import settings
from orchestrator_api.llm import (
    decide_image_ops,
    decide_tool_usage,
    generate_answer_with_llm,
    stream_answer_with_llm,
)
from orchestrator_api.rag.retrieve import retrieve_citations
from orchestrator_api.schemas import AnalysisResult, ChatRequest, ChatResponse, TraceInfo
from orchestrator_api.tools.mcp_tools import build_analyze_satellite_image_tool


async def run_chat_langchain(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    state: dict[str, Any] = {
        "request": request,
        "tools_used": [],
        "citations": [],
        "analysis": AnalysisResult(invoked=False),
        "rag_active": False,
        "rag_relaxed": False,
        "answer": "",
    }

    chain = (
        RunnableLambda(_route_step)
        | RunnableLambda(_rag_step)
        | RunnableLambda(_mcp_step)
        | RunnableLambda(_answer_step)
    )
    final_state = await chain.ainvoke(state)
    return _build_response(final_state, start=start)


async def run_chat_stream_langchain(request: ChatRequest) -> AsyncGenerator[dict, None]:
    start = time.perf_counter()
    state: dict[str, Any] = {
        "request": request,
        "tools_used": [],
        "citations": [],
        "analysis": AnalysisResult(invoked=False),
        "rag_active": False,
        "rag_relaxed": False,
        "answer": "",
    }

    yield {"type": "status", "stage": "route", "message": "deciding tool usage"}
    state = await _route_step(state)

    state = await _rag_step(state)
    yield {
        "type": "status",
        "stage": "rag",
        "used": state["rag_active"],
        "hits": len(state["citations"]),
        "min_score": settings.rag_min_score,
        "relaxed": state["rag_relaxed"],
    }

    state = await _mcp_step(state)
    analysis: AnalysisResult = state["analysis"]
    yield {
        "type": "status",
        "stage": "mcp",
        "invoked": analysis.invoked,
        "ops": [op.name for op in analysis.ops],
        "error": analysis.error,
    }

    yield {"type": "status", "stage": "llm", "message": "generating answer"}
    answer_parts: list[str] = []
    streamed = False
    async for token in stream_answer_with_llm(
        state["request"].question,
        state["citations"],
        state["analysis"],
    ):
        if not streamed:
            streamed = True
            state["tools_used"].append("llm.generate.stream")
            yield {"type": "answer_start"}
        answer_parts.append(token)
        yield {"type": "answer_chunk", "text": token}

    if streamed:
        state["answer"] = "".join(answer_parts)
    else:
        state = await _answer_step(state)
        yield {"type": "answer_start"}
        for chunk in _chunk_text(state["answer"], size=24):
            yield {"type": "answer_chunk", "text": chunk}

    response = _build_response(state, start=start)
    yield {"type": "final", "data": response.model_dump()}


async def _route_step(state: dict[str, Any]) -> dict[str, Any]:
    request: ChatRequest = state["request"]
    decision = await decide_tool_usage(
        question=request.question,
        image_available=bool(request.image_uri),
    )
    state["decision"] = decision
    state["tools_used"] = [
        "router.decision",
        f"route.rag:{str(decision.use_rag).lower()}",
        f"route.mcp:{str(decision.use_mcp).lower()}",
        f"route.reason:{decision.reason}",
    ]
    if decision.error:
        state["tools_used"].append(f"route.error:{decision.error}")
    return state


async def _rag_step(state: dict[str, Any]) -> dict[str, Any]:
    request: ChatRequest = state["request"]
    decision = state["decision"]
    citations: list = []
    rag_active = decision.use_rag
    rag_relaxed = False

    if rag_active and request.question.strip():
        citations = retrieve_citations(
            request.question,
            top_k=request.top_k,
            min_score=settings.rag_min_score,
        )
        if not citations:
            rag_relaxed = True
            citations = retrieve_citations(
                request.question,
                top_k=request.top_k,
                min_score=0.0,
            )

    if not decision.use_rag and not decision.use_mcp and request.question.strip():
        citations = retrieve_citations(
            request.question,
            top_k=request.top_k,
            min_score=settings.rag_min_score,
        )
        if citations:
            rag_active = True

    state["citations"] = citations
    state["rag_active"] = rag_active
    state["rag_relaxed"] = rag_relaxed
    if rag_active:
        state["tools_used"].append("rag.retrieve")
    if rag_relaxed:
        state["tools_used"].append("rag.retrieve.relaxed")
    return state


async def _mcp_step(state: dict[str, Any]) -> dict[str, Any]:
    request: ChatRequest = state["request"]
    decision = state["decision"]
    analysis = AnalysisResult(invoked=False)

    if decision.use_mcp and request.image_uri:
        state["tools_used"].append("mcp.analyze_satellite_image")
        if request.ops:
            ops = request.ops
            state["tools_used"].append("mcp.ops:client")
        else:
            ops, ops_reason = await decide_image_ops(request.question)
            state["tools_used"].append(f"mcp.ops:{ops_reason}")
        tool = build_analyze_satellite_image_tool()
        analysis = await tool.ainvoke(
            {"image_uri": request.image_uri, "ops": ops, "roi": request.roi}
        )

    state["analysis"] = analysis
    return state


async def _answer_step(state: dict[str, Any]) -> dict[str, Any]:
    request: ChatRequest = state["request"]
    citations = state["citations"]
    analysis = state["analysis"]

    answer, llm_error = await generate_answer_with_llm(request.question, citations, analysis)
    if answer:
        state["tools_used"].append("llm.generate")
        state["answer"] = answer
    else:
        if llm_error:
            state["tools_used"].append(f"llm.fallback:{llm_error}")
        state["answer"] = _compose_answer(request.question, citations, analysis)
    return state


def _build_response(state: dict[str, Any], start: float) -> ChatResponse:
    answer = _append_mcp_tools_to_answer(state["answer"], state["analysis"])
    latency_ms = int((time.perf_counter() - start) * 1000)
    trace = TraceInfo(tools=state["tools_used"], latency_ms=latency_ms)
    return ChatResponse(
        answer=answer,
        citations=state["citations"],
        analysis=state["analysis"],
        trace=trace,
    )


def _compose_answer(question: str, citations: list, analysis: AnalysisResult) -> str:
    title = f"질문: {question}" if question else "질문: (생략)"
    parts = [title]

    if citations:
        parts.append(f"RAG 근거 {len(citations)}건을 사용했습니다.")
    else:
        parts.append("RAG 근거를 사용하지 않았습니다.")

    if analysis.invoked and analysis.error:
        parts.append(f"영상 분석 실패: {analysis.error}")
    elif analysis.invoked and analysis.ops:
        op_names = ", ".join(op.name for op in analysis.ops)
        parts.append(f"영상 분석 실행: {op_names}")
    else:
        parts.append("영상 분석 미실행")

    return " ".join(parts)


def _append_mcp_tools_to_answer(answer: str, analysis: AnalysisResult) -> str:
    if not analysis.invoked:
        return answer
    if analysis.error:
        return f"{answer}\n\nMCP tools: (failed)"
    op_names = [op.name for op in analysis.ops if op.name]
    if not op_names:
        return f"{answer}\n\nMCP tools: (none)"
    return f"{answer}\n\nMCP tools: {', '.join(op_names)}"


def _chunk_text(text: str, size: int = 24) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]
