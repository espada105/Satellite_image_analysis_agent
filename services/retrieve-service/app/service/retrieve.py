from app.adapter.repository.retriever.neo4j import retrieve_graph_hints
from app.adapter.repository.retriever.pgvector import retrieve_citations
from app.common.config import TOP_K_DEFAULT
from packages.shared.app_shared.schema.chat import AnalysisResult, ChatRequest, ChatResponse, TraceInfo


def run_retrieve(request: ChatRequest) -> ChatResponse:
    top_k = request.top_k or TOP_K_DEFAULT
    citations = retrieve_citations(request.question, top_k)
    graph_hints = retrieve_graph_hints(request.question)

    answer = "검색 결과가 없습니다." if not citations else "요청하신 질의에 대한 검색 결과입니다."
    if graph_hints:
        answer += f" 그래프 힌트 {len(graph_hints)}건 반영"

    return ChatResponse(
        answer=answer,
        citations=citations,
        analysis=AnalysisResult(invoked=bool(request.image_uri)),
        trace=TraceInfo(tools=["pgvector", "neo4j"], latency_ms=10),
    )
