from orchestrator_api.rag.store import store
from orchestrator_api.schemas import Citation


def retrieve_citations(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[Citation]:
    results = store.search(query=query, top_k=max(top_k * 3, top_k), min_score=min_score)

    citations: list[Citation] = []
    chosen_ranges: dict[str, list[tuple[int, int]]] = {}
    for chunk, score in results:
        if _is_redundant(chunk.doc_id, chunk.line_start, chunk.line_end, chosen_ranges):
            continue

        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text[:220],
                score=round(score, 4),
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
        )
        chosen_ranges.setdefault(chunk.doc_id, []).append((chunk.line_start, chunk.line_end))
        if len(citations) >= top_k:
            break
    return citations


def _is_redundant(
    doc_id: str,
    line_start: int,
    line_end: int,
    chosen_ranges: dict[str, list[tuple[int, int]]],
) -> bool:
    for existing_start, existing_end in chosen_ranges.get(doc_id, []):
        overlap = min(line_end, existing_end) - max(line_start, existing_start) + 1
        if overlap <= 0:
            continue
        span = max(1, line_end - line_start + 1)
        overlap_ratio = overlap / span
        if overlap_ratio >= 0.6:
            return True
    return False
