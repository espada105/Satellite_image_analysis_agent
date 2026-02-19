from orchestrator_api.rag.embedder import embed_text
from orchestrator_api.rag.store import store
from orchestrator_api.schemas import Citation


def retrieve_citations(query: str, top_k: int = 3) -> list[Citation]:
    query_embedding = embed_text(query)
    results = store.search(query_embedding, top_k=top_k)

    citations: list[Citation] = []
    for chunk, score in results:
        if score <= 0:
            continue
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text[:220],
                score=round(score, 4),
            )
        )
    return citations
