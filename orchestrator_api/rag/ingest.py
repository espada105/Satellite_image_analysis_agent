from orchestrator_api.rag.chunker import chunk_text
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
            for chunk in chunks:
                chunk_id = f"{doc_id}:{chunk.line_start}-{chunk.line_end}"
                records.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=chunk.text,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                    )
                )
            store.add(records)
            ingested_count += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"document": doc_input, "error": str(exc)})

    return ingested_count, failures
