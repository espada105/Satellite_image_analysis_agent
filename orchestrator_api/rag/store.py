from collections import Counter
from dataclasses import dataclass
from threading import Lock

from orchestrator_api.rag.embedder import cosine_similarity


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    text: str
    embedding: Counter[str]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[ChunkRecord] = []
        self._lock = Lock()

    def add(self, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: Counter[str],
        top_k: int = 3,
    ) -> list[tuple[ChunkRecord, float]]:
        with self._lock:
            scored = [
                (chunk, cosine_similarity(query_embedding, chunk.embedding))
                for chunk in self._chunks
            ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()


store = InMemoryVectorStore()
