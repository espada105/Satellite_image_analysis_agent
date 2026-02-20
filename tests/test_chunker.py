from orchestrator_api.rag.chunker import chunk_text


def test_chunk_text_preserves_markdown_headers() -> None:
    text = "# Region A\ncloud analysis line\n## Change\nedge detection line"
    chunks = chunk_text(text, chunk_size=120, overlap=20)

    assert len(chunks) >= 2
    assert any("Section: Region A" in chunk.text for chunk in chunks)
    assert any("Section: Change" in chunk.text for chunk in chunks)


def test_chunk_text_keeps_original_line_offsets() -> None:
    text = "\n\n# Region A\nline a\nline b\n\n## Region B\nline c\n"
    chunks = chunk_text(text, chunk_size=40, overlap=8)

    assert chunks[0].line_start == 3
    assert chunks[0].line_end >= 4
    assert any(chunk.line_start == 7 for chunk in chunks)
