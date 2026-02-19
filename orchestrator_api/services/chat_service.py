import time
from collections.abc import AsyncGenerator

from orchestrator_api.config import settings
from orchestrator_api.llm import decide_tool_usage, generate_answer_with_llm
from orchestrator_api.mcp_client import analyze_image
from orchestrator_api.rag.retrieve import retrieve_citations
from orchestrator_api.schemas import AnalysisResult, ChatRequest, ChatResponse, TraceInfo


async def run_chat(request: ChatRequest) -> ChatResponse:
    response, _ = await _run_pipeline(request)
    return response


async def run_chat_stream(request: ChatRequest) -> AsyncGenerator[dict, None]:
    response, events = await _run_pipeline(request)
    for event in events:
        yield event
    yield {"type": "final", "data": response.model_dump()}


async def _run_pipeline(request: ChatRequest) -> tuple[ChatResponse, list[dict]]:
    start = time.perf_counter()
    events: list[dict] = []

    events.append({"type": "status", "stage": "route", "message": "deciding tool usage"})
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

    citations = []
    rag_active = decision.use_rag
    if rag_active and request.question.strip():
        tools_used.append("rag.retrieve")
        citations = retrieve_citations(
            request.question,
            top_k=request.top_k,
            min_score=settings.rag_min_score,
        )

    if not decision.use_rag and not decision.use_mcp and request.question.strip():
        tools_used.append("route.override:rag_probe")
        citations = retrieve_citations(
            request.question,
            top_k=request.top_k,
            min_score=settings.rag_min_score,
        )
        if citations:
            rag_active = True
            tools_used.append("rag.retrieve.probe_hit")
        else:
            tools_used.append("rag.retrieve.probe_miss")

    events.append(
        {
            "type": "status",
            "stage": "rag",
            "used": rag_active,
            "hits": len(citations),
            "min_score": settings.rag_min_score,
        }
    )

    analysis = AnalysisResult(invoked=False)
    if decision.use_mcp and request.image_uri:
        tools_used.append("mcp.analyze_satellite_image")
        ops = request.ops or ["edges", "cloud_mask_like", "masking_like"]
        analysis = await analyze_image(request.image_uri, ops=ops, roi=request.roi)

    events.append(
        {
            "type": "status",
            "stage": "mcp",
            "invoked": analysis.invoked,
            "ops": [op.name for op in analysis.ops],
            "error": analysis.error,
        }
    )

    events.append({"type": "status", "stage": "llm", "message": "generating answer"})
    answer, llm_error = await generate_answer_with_llm(request.question, citations, analysis)
    if answer:
        tools_used.append("llm.generate")
    else:
        if llm_error:
            tools_used.append(f"llm.fallback:{llm_error}")
        answer = _compose_answer(request.question, citations, analysis)

    latency_ms = int((time.perf_counter() - start) * 1000)
    trace = TraceInfo(tools=tools_used, latency_ms=latency_ms)

    response = ChatResponse(answer=answer, citations=citations, analysis=analysis, trace=trace)
    return response, events


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
