from orchestrator_api.rag.chunker import chunk_text
from orchestrator_api.rag.embedder import embed_text
from orchestrator_api.rag.parser import parse_document
from orchestrator_api.rag.store import ChunkRecord, store


def ingest_documents(
    documents: list[str],
    chunk_size: int = 500,
    overlap: int = 100,
) -> tuple[int, list[dict[str, str]]]:
    ingested_count = 0
    failures: list[dict[str, str]] = []

    for doc_input in documents:
        try:
            doc_id, text = parse_document(doc_input)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            records: list[ChunkRecord] = []
            for i, chunk in enumerate(chunks):
                records.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}:{i}",
                        text=chunk,
                        embedding=embed_text(chunk),
                    )
                )
            store.add(records)
            ingested_count += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"document": doc_input, "error": str(exc)})

    return ingested_count, failures
