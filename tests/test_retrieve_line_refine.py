from orchestrator_api.rag.retrieve import _refine_line_span


def test_refine_line_span_with_section_prefix() -> None:
    text = "Section: Q1\n## Q1\nNDVI = (NIR - Red) / (NIR + Red)\n구름 제거 후 계산 필요"
    start, end = _refine_line_span(
        query="NDVI 계산",
        text=text,
        chunk_line_start=80,
        chunk_line_end=95,
    )
    assert start == 81
    assert end == 81


def test_refine_line_span_falls_back_when_no_match() -> None:
    text = "Section: Q1\n## Q1\nalpha\nbeta"
    start, end = _refine_line_span(
        query="zeta",
        text=text,
        chunk_line_start=10,
        chunk_line_end=30,
    )
    assert start == 10
    assert end == 30
