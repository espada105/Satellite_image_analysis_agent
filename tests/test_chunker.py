from orchestrator_api.rag.chunker import chunk_text


def test_chunk_text_preserves_markdown_headers() -> None:
    text = "# Region A\ncloud analysis line\n## Change\nedge detection line"
    chunks = chunk_text(text, chunk_size=120, overlap=20)

    assert len(chunks) >= 2
    assert any("Section: Region A" in chunk.text for chunk in chunks)
    assert any("Section: Change" in chunk.text for chunk in chunks)
