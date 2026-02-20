import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
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
        self._chunk_ids: set[str] = set()
        self._postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._lexical_embeddings: list[Counter[str]] = []

        self._model = None
        self._tokenizer = None
        self._special_ids: set[int] = set()
        self._backend = "uninitialized"
        self._backend_error: str | None = None

        self._db_path = Path(settings.rag_store_db_path)
        if not self._db_path.is_absolute():
            self._db_path = Path.cwd() / self._db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._disk_loaded = False
        self._lock = Lock()
        self._init_db()

    def add(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        with self._lock:
            self._ensure_ready_locked()

            pending = [chunk for chunk in chunks if chunk.chunk_id not in self._chunk_ids]
            if not pending:
                return

            if self._backend == "sparse":
                self._add_sparse_locked(pending)
            else:
                self._add_lexical_locked(pending)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[tuple[ChunkRecord, float]]:
        if not query.strip():
            return []

        with self._lock:
            self._ensure_ready_locked()

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

    def clear(self, delete_disk: bool = True) -> None:
        with self._lock:
            self._records.clear()
            self._chunk_ids.clear()
            self._postings.clear()
            self._lexical_embeddings.clear()
            self._disk_loaded = False

            if delete_disk:
                with self._connect() as conn:
                    conn.execute("DELETE FROM postings")
                    conn.execute("DELETE FROM lexical")
                    conn.execute("DELETE FROM chunks")
                    conn.execute("DELETE FROM meta")
                    conn.commit()

    def count(self) -> int:
        with self._lock:
            self._ensure_ready_locked()
            return len(self._records)

    def backend_info(self) -> dict[str, str | None]:
        with self._lock:
            self._ensure_ready_locked()
            return {
                "backend": self._backend,
                "error": self._backend_error,
                "model": settings.rag_sparse_model,
                "db_path": str(self._db_path),
            }

    def _add_sparse_locked(self, pending: list[ChunkRecord]) -> None:
        texts = [chunk.text for chunk in pending]
        with torch.no_grad():
            doc_emb = self._model.encode_document(texts, batch_size=8)
        doc_dense = self._to_dense_numpy(doc_emb)

        chunk_rows: list[tuple[int, str, str, str, int, int]] = []
        posting_rows: list[tuple[int, int, float]] = []

        for chunk, vec in zip(pending, doc_dense, strict=True):
            doc_index = len(self._records)
            self._records.append(chunk)
            self._chunk_ids.add(chunk.chunk_id)
            chunk_rows.append(
                (
                    doc_index,
                    chunk.doc_id,
                    chunk.chunk_id,
                    chunk.text,
                    chunk.line_start,
                    chunk.line_end,
                )
            )

            nz = np.flatnonzero(vec > settings.rag_sparse_min_weight).tolist()
            nz = [token_id for token_id in nz if token_id not in self._special_ids]
            for token_id in nz:
                weight = float(vec[token_id])
                self._postings[token_id].append((doc_index, weight))
                posting_rows.append((token_id, doc_index, weight))

        with self._connect() as conn:
            conn.executemany(
                (
                    "INSERT INTO chunks(doc_idx, doc_id, chunk_id, text, line_start, line_end) "
                    "VALUES(?, ?, ?, ?, ?, ?)"
                ),
                chunk_rows,
            )
            conn.executemany(
                "INSERT INTO postings(token_id, doc_idx, weight) VALUES(?, ?, ?)",
                posting_rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('backend', ?)",
                (self._backend,),
            )
            conn.commit()

    def _add_lexical_locked(self, pending: list[ChunkRecord]) -> None:
        chunk_rows: list[tuple[int, str, str, str, int, int]] = []
        lexical_rows: list[tuple[int, str]] = []

        for chunk in pending:
            doc_index = len(self._records)
            self._records.append(chunk)
            self._chunk_ids.add(chunk.chunk_id)
            emb = embed_text(chunk.text)
            self._lexical_embeddings.append(emb)

            chunk_rows.append(
                (
                    doc_index,
                    chunk.doc_id,
                    chunk.chunk_id,
                    chunk.text,
                    chunk.line_start,
                    chunk.line_end,
                )
            )
            lexical_rows.append((doc_index, json.dumps(dict(emb))))

        with self._connect() as conn:
            conn.executemany(
                (
                    "INSERT INTO chunks(doc_idx, doc_id, chunk_id, text, line_start, line_end) "
                    "VALUES(?, ?, ?, ?, ?, ?)"
                ),
                chunk_rows,
            )
            conn.executemany(
                "INSERT INTO lexical(doc_idx, embedding_json) VALUES(?, ?)",
                lexical_rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('backend', ?)",
                (self._backend,),
            )
            conn.commit()

    def _ensure_ready_locked(self) -> None:
        self._ensure_backend_locked()
        if self._disk_loaded:
            return

        with self._connect() as conn:
            backend_row = conn.execute(
                "SELECT value FROM meta WHERE key='backend'"
            ).fetchone()
            stored_backend = backend_row[0] if backend_row else None

            if stored_backend and stored_backend != self._backend:
                conn.execute("DELETE FROM postings")
                conn.execute("DELETE FROM lexical")
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM meta")
                conn.commit()
                stored_backend = None

            rows = conn.execute(
                "SELECT doc_idx, doc_id, chunk_id, text, line_start, line_end "
                "FROM chunks ORDER BY doc_idx"
            ).fetchall()

            self._records = []
            self._chunk_ids = set()
            for _, doc_id, chunk_id, text, line_start, line_end in rows:
                record = ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    line_start=line_start,
                    line_end=line_end,
                )
                self._records.append(record)
                self._chunk_ids.add(chunk_id)

            if self._backend == "sparse":
                posting_rows = conn.execute(
                    "SELECT token_id, doc_idx, weight FROM postings"
                ).fetchall()
                self._postings = defaultdict(list)
                for token_id, doc_idx, weight in posting_rows:
                    self._postings[int(token_id)].append((int(doc_idx), float(weight)))
            else:
                lexical_rows = conn.execute(
                    "SELECT doc_idx, embedding_json FROM lexical ORDER BY doc_idx"
                ).fetchall()
                emb_map: dict[int, Counter[str]] = {}
                for doc_idx, emb_json in lexical_rows:
                    emb_map[int(doc_idx)] = Counter(json.loads(emb_json))
                self._lexical_embeddings = [
                    emb_map.get(i, Counter()) for i in range(len(self._records))
                ]

            if stored_backend is None:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('backend', ?)",
                    (self._backend,),
                )
                conn.commit()

        self._disk_loaded = True

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

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_idx INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS postings (
                    token_id INTEGER NOT NULL,
                    doc_idx INTEGER NOT NULL,
                    weight REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_postings_token ON postings(token_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lexical (
                    doc_idx INTEGER PRIMARY KEY,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    @staticmethod
    def _to_dense_numpy(value) -> np.ndarray:
        if hasattr(value, "to_dense"):
            return value.to_dense().float().cpu().numpy()
        if isinstance(value, torch.Tensor):
            return value.float().cpu().numpy()
        return np.asarray(value)


store = SparseVectorStore()
