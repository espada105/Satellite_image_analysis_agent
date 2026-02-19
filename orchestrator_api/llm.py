import json

import httpx

from orchestrator_api.config import settings
from orchestrator_api.schemas import AnalysisResult, Citation

SYSTEM_PROMPT = (
    "You are a satellite image analysis assistant. "
    "Use citations and analysis results when provided. "
    "If they are insufficient, still answer the user's general question safely and concisely."
)


def llm_enabled() -> bool:
    return bool(settings.llm_api_key and settings.llm_model)


async def generate_answer_with_llm(
    question: str,
    citations: list[Citation],
    analysis: AnalysisResult,
    timeout_s: float = 20.0,
) -> tuple[str | None, str | None]:
    if not llm_enabled():
        return None, "llm_not_configured"

    context = {
        "question": question,
        "citations": [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "snippet": c.snippet,
                "score": c.score,
            }
            for c in citations
        ],
        "analysis": analysis.model_dump(),
    }

    responses_body = {
        "model": settings.llm_model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(context, ensure_ascii=False),
                    }
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            responses_url = f"{settings.llm_base_url.rstrip('/')}/responses"
            response = await client.post(responses_url, headers=headers, json=responses_body)
            if response.is_success:
                parsed = _parse_responses_output(response.json())
                if parsed:
                    return parsed, None

            # Fallback for compatibility with providers/configs expecting chat completions.
            chat_body = {
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
            }
            chat_url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
            chat_response = await client.post(chat_url, headers=headers, json=chat_body)
            chat_response.raise_for_status()
            parsed_chat = _parse_chat_completions_output(chat_response.json())
            if parsed_chat:
                return parsed_chat, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    return None, "empty_llm_output"


def _parse_responses_output(data: dict) -> str | None:
    output_text = data.get("output_text")
    if output_text:
        return output_text.strip()

    output_items = data.get("output", [])
    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"].strip()

    return None


def _parse_chat_completions_output(data: dict) -> str | None:
    choices = data.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    text = message.get("content")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None
