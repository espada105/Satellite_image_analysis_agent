from pathlib import Path

from orchestrator_api.rag.ingest import ingest_documents
from orchestrator_api.rag.langchain_retriever import ExistingStoreRetriever
from orchestrator_api.rag.store import store


def test_existing_store_retriever_wraps_custom_store(tmp_path: Path) -> None:
    store.clear()
    doc = tmp_path / "doc.txt"
    doc.write_text("Cloud mask and change detection workflow", encoding="utf-8")
    count, failed = ingest_documents([str(doc)])
    assert count == 1
    assert failed == []

    retriever = ExistingStoreRetriever(top_k=2, min_score=0.0)
    docs = retriever.invoke("cloud detection")

    assert docs
    metadata = docs[0].metadata
    assert metadata["doc_id"] == str(doc)
    assert ":" in metadata["chunk_id"]
    assert isinstance(metadata["line_start"], int)
    assert isinstance(metadata["line_end"], int)
    assert isinstance(metadata["score"], float)
