import re

from orchestrator_api.rag.store import store
from orchestrator_api.schemas import Citation

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9가-힣_]+")


def retrieve_citations(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[Citation]:
    results = store.search(query=query, top_k=max(top_k * 3, top_k), min_score=min_score)

    citations: list[Citation] = []
    chosen_ranges: dict[str, list[tuple[int, int]]] = {}
    for chunk, score in results:
        refined_start, refined_end = _refine_line_span(query, chunk.text, chunk.line_start, chunk.line_end)
        if _is_redundant(chunk.doc_id, refined_start, refined_end, chosen_ranges):
            continue

        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text[:220],
                score=round(score, 4),
                line_start=refined_start,
                line_end=refined_end,
            )
        )
        chosen_ranges.setdefault(chunk.doc_id, []).append((refined_start, refined_end))
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


def _refine_line_span(query: str, text: str, chunk_line_start: int, chunk_line_end: int) -> tuple[int, int]:
    lines = text.splitlines()
    if not lines:
        return chunk_line_start, chunk_line_end

    source_offset = chunk_line_start
    if lines and lines[0].startswith("Section: "):
        lines = lines[1:]
        source_offset += 1
    if not lines:
        return chunk_line_start, chunk_line_end

    query_terms = [token.lower() for token in TOKEN_PATTERN.findall(query) if len(token) >= 2]
    if not query_terms:
        return chunk_line_start, chunk_line_end

    per_line_scores = []
    for line in lines:
        lowered = line.lower()
        score = sum(1 for term in query_terms if term in lowered)
        per_line_scores.append(score)

    max_score = max(per_line_scores, default=0)
    if max_score <= 0:
        return chunk_line_start, chunk_line_end

    best_idx = per_line_scores.index(max_score)
    left = best_idx
    right = best_idx

    while left - 1 >= 0 and per_line_scores[left - 1] > 0:
        left -= 1
    while right + 1 < len(per_line_scores) and per_line_scores[right + 1] > 0:
        right += 1

    line_start = source_offset + left
    line_end = source_offset + right
    line_start = max(chunk_line_start, line_start)
    line_end = min(chunk_line_end, line_end)
    if line_start > line_end:
        return chunk_line_start, chunk_line_end
    return line_start, line_end
