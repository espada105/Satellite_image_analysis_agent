from packages.shared.app_shared.schema.chat import Citation


def retrieve_citations(question: str, top_k: int) -> list[Citation]:
    if not question.strip():
        return []
    return [
        Citation(
            doc_id="stub-doc",
            chunk_id="chunk-1",
            snippet=f"retrieved context for: {question[:60]}",
            score=0.9,
            line_start=1,
            line_end=3,
        )
    ][:top_k]
