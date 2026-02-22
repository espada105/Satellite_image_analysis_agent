import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mcp_base_url: str = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8100")
    rag_index_name: str = os.getenv("RAG_INDEX_NAME", "default")
    rag_store_db_path: str = os.getenv("RAG_STORE_DB_PATH", "data/vector_store/rag_store.sqlite3")
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.05"))
    rag_sparse_model: str = os.getenv("RAG_SPARSE_MODEL", "telepix/PIXIE-Splade-v1.0")
    rag_sparse_min_weight: float = float(os.getenv("RAG_SPARSE_MIN_WEIGHT", "0.01"))
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_json: bool = os.getenv("LOG_JSON", "true").lower() == "true"
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    rag_hybrid_alpha: float = float(os.getenv("RAG_HYBRID_ALPHA", "0.65"))
    rag_bm25_k1: float = float(os.getenv("RAG_BM25_K1", "1.2"))
    rag_bm25_b: float = float(os.getenv("RAG_BM25_B", "0.75"))
    rag_rerank_heading_boost: float = float(os.getenv("RAG_RERANK_HEADING_BOOST", "0.15"))
    use_langchain_pipeline: bool = os.getenv("USE_LANGCHAIN_PIPELINE", "true").lower() == "true"


settings = Settings()


def get_verified_user_ids() -> set[str]:
    raw = os.getenv("VERIFIED_USER_IDS", "")
    return {user_id.strip() for user_id in raw.split(",") if user_id.strip()}
