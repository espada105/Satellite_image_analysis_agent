from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from sentence_transformers import SparseEncoder
from transformers import AutoTokenizer

from orchestrator_api.config import settings
from orchestrator_api.rag.embedder import cosine_similarity, embed_text


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    text: str
    line_start: int
    line_end: int


class SparseVectorStore:
    def __init__(self) -> None:
        self._records: list[ChunkRecord] = []
        self._postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._lexical_embeddings: list[Counter[str]] = []

        self._model = None
        self._tokenizer = None
        self._special_ids: set[int] = set()
        self._backend = "uninitialized"
        self._backend_error: str | None = None
        self._lock = Lock()

    def add(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        with self._lock:
            self._ensure_backend_locked()

            if self._backend == "sparse":
                texts = [chunk.text for chunk in chunks]
                with torch.no_grad():
                    doc_emb = self._model.encode_document(texts, batch_size=8)
                doc_dense = self._to_dense_numpy(doc_emb)

                for chunk, vec in zip(chunks, doc_dense, strict=True):
                    doc_index = len(self._records)
                    self._records.append(chunk)

                    nz = np.flatnonzero(vec > settings.rag_sparse_min_weight).tolist()
                    nz = [token_id for token_id in nz if token_id not in self._special_ids]
                    for token_id in nz:
                        self._postings[token_id].append((doc_index, float(vec[token_id])))
            else:
                for chunk in chunks:
                    self._records.append(chunk)
                    self._lexical_embeddings.append(embed_text(chunk.text))

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[tuple[ChunkRecord, float]]:
        if not query.strip():
            return []

        with self._lock:
            self._ensure_backend_locked()

            if self._backend == "sparse":
                with torch.no_grad():
                    q_vec = self._model.encode_query(query)
                q_dense = self._to_dense_numpy(q_vec).ravel()

                q_nz = np.flatnonzero(q_dense > settings.rag_sparse_min_weight).tolist()
                q_nz = [token_id for token_id in q_nz if token_id not in self._special_ids]

                scores: dict[int, float] = defaultdict(float)
                for token_id in q_nz:
                    q_weight = float(q_dense[token_id])
                    for doc_idx, d_weight in self._postings.get(token_id, []):
                        scores[doc_idx] += q_weight * d_weight

                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                results: list[tuple[ChunkRecord, float]] = []
                for doc_idx, score in ranked:
                    if score < min_score:
                        continue
                    results.append((self._records[doc_idx], float(score)))
                    if len(results) >= top_k:
                        break
                return results

            query_embedding = embed_text(query)
            scored = [
                (chunk, cosine_similarity(query_embedding, embedding))
                for chunk, embedding in zip(self._records, self._lexical_embeddings, strict=True)
            ]
            scored = [item for item in scored if item[1] >= min_score]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_k]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._postings.clear()
            self._lexical_embeddings.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def backend_info(self) -> dict[str, str | None]:
        with self._lock:
            self._ensure_backend_locked()
            return {
                "backend": self._backend,
                "error": self._backend_error,
                "model": settings.rag_sparse_model,
            }

    def _ensure_backend_locked(self) -> None:
        if self._backend != "uninitialized":
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SparseEncoder(settings.rag_sparse_model).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(settings.rag_sparse_model)
            self._special_ids = set(getattr(self._tokenizer, "all_special_ids", []) or [])
            self._backend = "sparse"
        except Exception as exc:  # noqa: BLE001
            self._backend = "lexical"
            self._backend_error = str(exc)

    @staticmethod
    def _to_dense_numpy(value) -> np.ndarray:
        if hasattr(value, "to_dense"):
            return value.to_dense().float().cpu().numpy()
        if isinstance(value, torch.Tensor):
            return value.float().cpu().numpy()
        return np.asarray(value)


store = SparseVectorStore()
