from orchestrator_api.rag.store import store
from orchestrator_api.schemas import Citation


def retrieve_citations(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[Citation]:
    results = store.search(query=query, top_k=top_k, min_score=min_score)

    citations: list[Citation] = []
    for chunk, score in results:
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text[:220],
                score=round(score, 4),
            )
        )
    return citations
