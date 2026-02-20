import time
from collections.abc import AsyncGenerator

from orchestrator_api.config import settings
from orchestrator_api.llm import decide_image_ops, decide_tool_usage, generate_answer_with_llm, stream_answer_with_llm
from orchestrator_api.mcp_client import analyze_image
from orchestrator_api.rag.retrieve import retrieve_citations
from orchestrator_api.schemas import AnalysisResult, ChatRequest, ChatResponse, TraceInfo


async def run_chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()

    decision = await decide_tool_usage(
        question=request.question,
        image_available=bool(request.image_uri),
    )

    tools_used = [
        "router.decision",
        f"route.rag:{str(decision.use_rag).lower()}",
        f"route.mcp:{str(decision.use_mcp).lower()}",
        f"route.reason:{decision.reason}",
    ]
    if decision.error:
        tools_used.append(f"route.error:{decision.error}")

    citations, rag_relaxed, rag_active = _run_rag(request, decision.use_rag, decision.use_mcp)
    if rag_active:
        tools_used.append("rag.retrieve")
    if rag_relaxed:
        tools_used.append("rag.retrieve.relaxed")

    analysis = AnalysisResult(invoked=False)
    if decision.use_mcp and request.image_uri:
        tools_used.append("mcp.analyze_satellite_image")
        if request.ops:
            ops = request.ops
            tools_used.append("mcp.ops:client")
        else:
            ops, ops_reason = await decide_image_ops(request.question)
            tools_used.append(f"mcp.ops:{ops_reason}")
        analysis = await analyze_image(request.image_uri, ops=ops, roi=request.roi)

    answer, llm_error = await generate_answer_with_llm(request.question, citations, analysis)
    if answer:
        tools_used.append("llm.generate")
    else:
        if llm_error:
            tools_used.append(f"llm.fallback:{llm_error}")
        answer = _compose_answer(request.question, citations, analysis)
    answer = _append_mcp_tools_to_answer(answer, analysis)

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace = TraceInfo(tools=tools_used, latency_ms=latency_ms)
    return ChatResponse(answer=answer, citations=citations, analysis=analysis, trace=trace)


async def run_chat_stream(request: ChatRequest) -> AsyncGenerator[dict, None]:
    start = time.perf_counter()

    yield {"type": "status", "stage": "route", "message": "deciding tool usage"}
    decision = await decide_tool_usage(
        question=request.question,
        image_available=bool(request.image_uri),
    )

    tools_used = [
        "router.decision",
        f"route.rag:{str(decision.use_rag).lower()}",
        f"route.mcp:{str(decision.use_mcp).lower()}",
        f"route.reason:{decision.reason}",
    ]
    if decision.error:
        tools_used.append(f"route.error:{decision.error}")

    citations, rag_relaxed, rag_active = _run_rag(request, decision.use_rag, decision.use_mcp)
    if rag_active:
        tools_used.append("rag.retrieve")
    if rag_relaxed:
        tools_used.append("rag.retrieve.relaxed")

    yield {
        "type": "status",
        "stage": "rag",
        "used": rag_active,
        "hits": len(citations),
        "min_score": settings.rag_min_score,
        "relaxed": rag_relaxed,
    }

    analysis = AnalysisResult(invoked=False)
    if decision.use_mcp and request.image_uri:
        tools_used.append("mcp.analyze_satellite_image")
        if request.ops:
            ops = request.ops
            tools_used.append("mcp.ops:client")
        else:
            ops, ops_reason = await decide_image_ops(request.question)
            tools_used.append(f"mcp.ops:{ops_reason}")
        analysis = await analyze_image(request.image_uri, ops=ops, roi=request.roi)

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
    async for token in stream_answer_with_llm(request.question, citations, analysis):
        if not streamed:
            streamed = True
            tools_used.append("llm.generate.stream")
            yield {"type": "answer_start"}
        answer_parts.append(token)
        yield {"type": "answer_chunk", "text": token}

    if streamed:
        answer = "".join(answer_parts)
    else:
        answer, llm_error = await generate_answer_with_llm(request.question, citations, analysis)
        if answer:
            tools_used.append("llm.generate")
            yield {"type": "answer_start"}
            for chunk in _chunk_text(answer, size=24):
                yield {"type": "answer_chunk", "text": chunk}
        else:
            if llm_error:
                tools_used.append(f"llm.fallback:{llm_error}")
            answer = _compose_answer(request.question, citations, analysis)
            yield {"type": "answer_start"}
            for chunk in _chunk_text(answer, size=24):
                yield {"type": "answer_chunk", "text": chunk}
    answer = _append_mcp_tools_to_answer(answer, analysis)

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace = TraceInfo(tools=tools_used, latency_ms=latency_ms)
    response = ChatResponse(answer=answer, citations=citations, analysis=analysis, trace=trace)
    yield {"type": "final", "data": response.model_dump()}


def _run_rag(
    request: ChatRequest,
    decision_use_rag: bool,
    decision_use_mcp: bool,
) -> tuple[list, bool, bool]:
    citations = []
    rag_active = decision_use_rag
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

    if not decision_use_rag and not decision_use_mcp and request.question.strip():
        citations = retrieve_citations(
            request.question,
            top_k=request.top_k,
            min_score=settings.rag_min_score,
        )
        if citations:
            rag_active = True

    return citations, rag_relaxed, rag_active


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


def _chunk_text(text: str, size: int = 24) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _append_mcp_tools_to_answer(answer: str, analysis: AnalysisResult) -> str:
    if not analysis.invoked:
        return answer
    if analysis.error:
        return f"{answer}\n\nMCP tools: (failed)"
    op_names = [op.name for op in analysis.ops if op.name]
    if not op_names:
        return f"{answer}\n\nMCP tools: (none)"
    return f"{answer}\n\nMCP tools: {', '.join(op_names)}"
