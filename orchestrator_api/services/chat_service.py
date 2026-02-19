import time

from orchestrator_api.llm import generate_answer_with_llm
from orchestrator_api.mcp_client import analyze_image
from orchestrator_api.rag.retrieve import retrieve_citations
from orchestrator_api.router import should_invoke_mcp
from orchestrator_api.schemas import AnalysisResult, ChatRequest, ChatResponse, TraceInfo


async def run_chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    tools_used = ["rag.retrieve"]

    citations = retrieve_citations(request.question, top_k=request.top_k)

    analysis = AnalysisResult(invoked=False)
    if should_invoke_mcp(request.question, request.image_uri):
        tools_used.append("mcp.analyze_satellite_image")
        ops = request.ops or ["edges", "cloud_mask_like"]
        analysis = await analyze_image(request.image_uri or "", ops=ops, roi=request.roi)

    answer, llm_error = await generate_answer_with_llm(request.question, citations, analysis)
    if answer:
        tools_used.append("llm.generate")
    else:
        if llm_error:
            tools_used.append(f"llm.fallback:{llm_error}")
        answer = _compose_answer(request.question, citations, analysis)
    latency_ms = int((time.perf_counter() - start) * 1000)
    trace = TraceInfo(tools=tools_used, latency_ms=latency_ms)

    return ChatResponse(answer=answer, citations=citations, analysis=analysis, trace=trace)


def _compose_answer(question: str, citations: list, analysis: AnalysisResult) -> str:
    parts = [f"질문: {question}"]

    if citations:
        parts.append("RAG 근거를 바탕으로 관련 문서를 찾았습니다.")
    else:
        parts.append("RAG에서 직접적으로 일치하는 근거를 찾지 못했습니다.")

    if analysis.invoked and analysis.error:
        parts.append(f"영상 분석은 실패했습니다: {analysis.error}")
    elif analysis.invoked and analysis.ops:
        summaries = "; ".join(f"{op.name}: {op.summary}" for op in analysis.ops)
        parts.append(f"영상 분석 요약: {summaries}")
    elif analysis.invoked:
        parts.append("영상 분석을 호출했지만 유의미한 결과가 없었습니다.")

    return " ".join(parts)
