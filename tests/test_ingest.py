from pathlib import Path

from orchestrator_api.rag.ingest import ingest_documents
from orchestrator_api.rag.retrieve import retrieve_citations
from orchestrator_api.rag.store import store


def test_ingest_and_retrieve(tmp_path: Path) -> None:
    store.clear()
    doc = tmp_path / "doc1.txt"
    doc.write_text("Sentinel-2 cloud mask workflow and change detection guide", encoding="utf-8")

    count, failed = ingest_documents([str(doc)])

    assert count == 1
    assert failed == []

    cites = retrieve_citations("cloud change", top_k=3)
    assert len(cites) >= 1
    assert "cloud" in cites[0].snippet.lower() or "change" in cites[0].snippet.lower()
    assert cites[0].line_start is not None
    assert cites[0].line_end is not None
    assert ":" in cites[0].chunk_id and "-" in cites[0].chunk_id
