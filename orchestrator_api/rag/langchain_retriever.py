from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from orchestrator_api.rag.store import store


class ExistingStoreRetriever(BaseRetriever):
    """LangChain retriever adapter over the existing custom store/search model."""

    top_k: int = 3
    min_score: float = 0.0
    search_multiplier: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        if not query.strip():
            return []

        raw_hits = store.search(
            query=query,
            top_k=max(self.top_k * self.search_multiplier, self.top_k),
            min_score=self.min_score,
        )

        docs: list[Document] = []
        for chunk, score in raw_hits:
            docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "score": float(score),
                    },
                )
            )
        return docs
