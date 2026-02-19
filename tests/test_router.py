from orchestrator_api.router import should_invoke_mcp


def test_router_invokes_mcp_with_keyword_and_image() -> None:
    assert should_invoke_mcp("이 지역 변화 탐지 해줘", "/tmp/a.png") is True


def test_router_skips_without_image() -> None:
    assert should_invoke_mcp("변화 탐지 해줘", None) is False


def test_router_invokes_when_image_exists_even_without_visual_keyword() -> None:
    assert should_invoke_mcp("센서 메타데이터를 설명해줘", "/tmp/a.png") is True
