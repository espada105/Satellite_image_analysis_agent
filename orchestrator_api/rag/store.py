import json
import math
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
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avg_doc_length: float = 0.0

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
            query_embedding = embed_text(query)
            bm25_scores = self._compute_bm25_scores(query_embedding)
            semantic_scores: dict[int, float] = defaultdict(float)

            if self._backend == "sparse":
                with torch.no_grad():
                    q_vec = self._model.encode_query(query)
                q_dense = self._to_dense_numpy(q_vec).ravel()

                q_nz = np.flatnonzero(q_dense > settings.rag_sparse_min_weight).tolist()
                q_nz = [token_id for token_id in q_nz if token_id not in self._special_ids]

                for token_id in q_nz:
                    q_weight = float(q_dense[token_id])
                    for doc_idx, d_weight in self._postings.get(token_id, []):
                        semantic_scores[doc_idx] += q_weight * d_weight
            else:
                for idx, emb in enumerate(self._lexical_embeddings):
                    semantic_scores[idx] = cosine_similarity(query_embedding, emb)

            max_sem = max(semantic_scores.values(), default=0.0) or 1.0
            max_bm25 = max(bm25_scores.values(), default=0.0) or 1.0
            alpha = max(0.0, min(1.0, settings.rag_hybrid_alpha))

            merged: list[tuple[ChunkRecord, float]] = []
            for doc_idx, chunk in enumerate(self._records):
                sem = semantic_scores.get(doc_idx, 0.0) / max_sem
                lex = bm25_scores.get(doc_idx, 0.0) / max_bm25
                score = alpha * sem + (1.0 - alpha) * lex
                score += self._rerank_score_boost(query_embedding, chunk.text)
                if score < min_score:
                    continue
                merged.append((chunk, float(score)))

            merged.sort(key=lambda item: item[1], reverse=True)
            return merged[:top_k]

    def clear(self, delete_disk: bool = True) -> None:
        with self._lock:
            # Recreate schema defensively in case test bootstrap removed the DB file.
            self._init_db()
            self._records.clear()
            self._chunk_ids.clear()
            self._postings.clear()
            self._lexical_embeddings.clear()
            self._doc_term_freqs.clear()
            self._doc_lengths.clear()
            self._doc_freqs.clear()
            self._avg_doc_length = 0.0
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
            tf = embed_text(chunk.text)
            self._append_doc_stats(tf)
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
            self._append_doc_stats(emb)

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

        # Test bootstrap may recreate/remove sqlite file between imports.
        self._init_db()
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

        self._rebuild_doc_stats_locked()
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

    def _append_doc_stats(self, tf: Counter[str]) -> None:
        self._doc_term_freqs.append(tf)
        doc_len = sum(tf.values())
        self._doc_lengths.append(doc_len)
        for term in tf:
            self._doc_freqs[term] += 1
        total_docs = len(self._doc_lengths)
        self._avg_doc_length = (sum(self._doc_lengths) / total_docs) if total_docs else 0.0

    def _rebuild_doc_stats_locked(self) -> None:
        self._doc_term_freqs = []
        self._doc_lengths = []
        self._doc_freqs = Counter()
        for chunk in self._records:
            tf = embed_text(chunk.text)
            self._doc_term_freqs.append(tf)
            self._doc_lengths.append(sum(tf.values()))
            for term in tf:
                self._doc_freqs[term] += 1
        total_docs = len(self._doc_lengths)
        self._avg_doc_length = (sum(self._doc_lengths) / total_docs) if total_docs else 0.0

    def _compute_bm25_scores(self, query_tf: Counter[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        if not query_tf or not self._doc_term_freqs:
            return scores

        total_docs = len(self._doc_term_freqs)
        avgdl = self._avg_doc_length or 1.0
        k1 = settings.rag_bm25_k1
        b = settings.rag_bm25_b

        for term in query_tf:
            df = self._doc_freqs.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            for idx, doc_tf in enumerate(self._doc_term_freqs):
                tf = doc_tf.get(term, 0)
                if tf <= 0:
                    continue
                dl = self._doc_lengths[idx] or 1
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                scores[idx] += idf * ((tf * (k1 + 1.0)) / denom)

        return scores

    def _rerank_score_boost(self, query_tf: Counter[str], text: str) -> float:
        if not query_tf or not text:
            return 0.0
        first_line = text.splitlines()[0].lower() if text else ""
        coverage = 0
        for term in query_tf:
            if term.lower() in first_line:
                coverage += 1
        if coverage == 0:
            return 0.0
        ratio = coverage / max(1, len(query_tf))
        return ratio * settings.rag_rerank_heading_boost

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
