import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mcp_base_url: str = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8100")
    rag_index_name: str = os.getenv("RAG_INDEX_NAME", "default")
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.05"))
    rag_sparse_model: str = os.getenv("RAG_SPARSE_MODEL", "telepix/PIXIE-Splade-v1.0")
    rag_sparse_min_weight: float = float(os.getenv("RAG_SPARSE_MIN_WEIGHT", "0.01"))
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")


settings = Settings()


def get_verified_user_ids() -> set[str]:
    raw = os.getenv("VERIFIED_USER_IDS", "")
    return {user_id.strip() for user_id in raw.split(",") if user_id.strip()}
