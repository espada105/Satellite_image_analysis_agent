import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    _LANGCHAIN_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001
    ChatPromptTemplate = None  # type: ignore[assignment]
    ChatOpenAI = None  # type: ignore[assignment]
    _LANGCHAIN_IMPORT_ERROR = str(exc)

from orchestrator_api.config import settings
from orchestrator_api.schemas import AnalysisResult, Citation

SYSTEM_PROMPT = (
    "You are a satellite image analysis assistant. "
    "Use citations and analysis results when provided. "
    "If they are insufficient, still answer the user's general question safely and concisely. "
    "Do not invent URLs or markdown image links. "
    "Do not claim files that are not in provided analysis artifacts."
)

ROUTER_PROMPT = (
    "Decide tool usage for a satellite assistant. Return ONLY JSON with keys "
    "use_rag (bool), use_mcp (bool), reason (string). "
    "Rules: use_mcp requires image_available=true. "
    "If question asks knowledge/explanation/document-backed info, use_rag=true. "
    "If question is visual analysis and image is available, use_mcp=true."
)

OPS_SELECTOR_PROMPT = (
    "Choose image analysis ops for a satellite assistant. Return ONLY JSON with keys "
    "ops (array of strings), reason (string). "
    "Allowed ops: edges, threshold, morphology, cloud_mask_like, masking_like. "
    "Pick 1-3 ops that best match the user question and image-analysis intent. "
    "Do not include any key except ops and reason."
)

SUPPORTED_OPS = {"edges", "threshold", "morphology", "cloud_mask_like", "masking_like"}


@dataclass(frozen=True)
class ToolDecision:
    use_rag: bool
    use_mcp: bool
    reason: str = ""
    error: str | None = None


class RouterOutput(BaseModel):
    use_rag: bool
    use_mcp: bool
    reason: str = Field(default="llm_router")


class OpsSelectorOutput(BaseModel):
    ops: list[str] = Field(default_factory=list)
    reason: str = Field(default="llm_ops_selector")


def llm_enabled() -> bool:
    return bool(settings.llm_api_key and settings.llm_model)


def langchain_enabled() -> bool:
    return llm_enabled() and _LANGCHAIN_IMPORT_ERROR is None


async def decide_tool_usage(
    question: str,
    image_available: bool,
    timeout_s: float = 12.0,
) -> ToolDecision:
    fallback = ToolDecision(
        use_rag=bool(question.strip()),
        use_mcp=image_available,
        reason="rule_fallback",
    )
    if not llm_enabled():
        return ToolDecision(
            use_rag=fallback.use_rag,
            use_mcp=fallback.use_mcp,
            reason=fallback.reason,
            error="llm_not_configured",
        )

    if langchain_enabled():
        try:
            chain = _router_chain(timeout_s=timeout_s)
            parsed: RouterOutput = await chain.ainvoke(
                {"question": question, "image_available": str(image_available).lower()}
            )
            return ToolDecision(
                use_rag=bool(parsed.use_rag),
                use_mcp=bool(parsed.use_mcp) and image_available,
                reason=str(parsed.reason or "llm_router"),
            )
        except Exception as exc:  # noqa: BLE001
            legacy = await _legacy_json_prompt_call(
                system_prompt=ROUTER_PROMPT,
                user_payload={"question": question, "image_available": image_available},
                timeout_s=timeout_s,
            )
            if legacy:
                return ToolDecision(
                    use_rag=bool(legacy.get("use_rag", fallback.use_rag)),
                    use_mcp=bool(legacy.get("use_mcp", fallback.use_mcp)) and image_available,
                    reason=str(legacy.get("reason", "llm_router")),
                    error=f"langchain_failed:{exc}",
                )
            return ToolDecision(
                use_rag=fallback.use_rag,
                use_mcp=fallback.use_mcp,
                reason=fallback.reason,
                error=f"langchain_failed:{exc}",
            )

    try:
        parsed = await _legacy_json_prompt_call(
            system_prompt=ROUTER_PROMPT,
            user_payload={"question": question, "image_available": image_available},
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolDecision(
            use_rag=fallback.use_rag,
            use_mcp=fallback.use_mcp,
            reason=fallback.reason,
            error=str(exc),
        )

    if not parsed:
        return ToolDecision(
            use_rag=fallback.use_rag,
            use_mcp=fallback.use_mcp,
            reason=fallback.reason,
            error="router_parse_failed",
        )

    use_rag = bool(parsed.get("use_rag", fallback.use_rag))
    use_mcp = bool(parsed.get("use_mcp", fallback.use_mcp)) and image_available
    reason = str(parsed.get("reason", "llm_router"))
    return ToolDecision(use_rag=use_rag, use_mcp=use_mcp, reason=reason)


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
        "analysis": _analysis_for_llm(analysis),
    }

    if langchain_enabled():
        try:
            chain = _answer_chain(timeout_s=timeout_s)
            payload = json.dumps(context, ensure_ascii=False)
            response = await chain.ainvoke({"payload": payload})
            text = _extract_text_from_langchain_message(response)
            if text:
                return text, None
        except Exception as exc:  # noqa: BLE001
            legacy = await _legacy_text_prompt_call(
                system_prompt=SYSTEM_PROMPT,
                user_payload=context,
                timeout_s=timeout_s,
            )
            if legacy:
                return legacy.strip(), f"langchain_failed:{exc}"
            return None, str(exc)

    try:
        response_text = await _legacy_text_prompt_call(
            system_prompt=SYSTEM_PROMPT,
            user_payload=context,
            timeout_s=timeout_s,
        )
        if response_text and response_text.strip():
            return response_text.strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    return None, "empty_llm_output"


async def decide_image_ops(
    question: str,
    timeout_s: float = 12.0,
) -> tuple[list[str], str]:
    fallback_ops = _fallback_ops(question)
    if not llm_enabled():
        return fallback_ops, "ops_rule_fallback:llm_not_configured"

    selector_input = {"question": question, "allowed_ops": sorted(SUPPORTED_OPS)}

    if langchain_enabled():
        try:
            chain = _ops_selector_chain(timeout_s=timeout_s)
            parsed: OpsSelectorOutput = await chain.ainvoke(
                {
                    "question": selector_input["question"],
                    "allowed_ops": ", ".join(selector_input["allowed_ops"]),
                }
            )
            chosen_ops = _normalize_selected_ops(parsed.ops)
            if not chosen_ops:
                return fallback_ops, "ops_rule_fallback:selector_empty"
            reason = str(parsed.reason or "llm_ops_selector")
            return chosen_ops, reason
        except Exception as exc:  # noqa: BLE001
            legacy = await _legacy_json_prompt_call(
                system_prompt=OPS_SELECTOR_PROMPT,
                user_payload=selector_input,
                timeout_s=timeout_s,
            )
            if legacy:
                chosen_ops = _normalize_selected_ops(legacy.get("ops", []))
                if chosen_ops:
                    return chosen_ops, f"{legacy.get('reason', 'llm_ops_selector')}:legacy"
            return fallback_ops, f"ops_rule_fallback:{exc}"

    try:
        parsed = await _legacy_json_prompt_call(
            system_prompt=OPS_SELECTOR_PROMPT,
            user_payload=selector_input,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        return fallback_ops, f"ops_rule_fallback:{exc}"

    if not parsed:
        return fallback_ops, "ops_rule_fallback:selector_parse_failed"

    raw_ops = parsed.get("ops", [])
    if not isinstance(raw_ops, list):
        return fallback_ops, "ops_rule_fallback:selector_invalid_type"

    chosen_ops = _normalize_selected_ops(raw_ops)
    if not chosen_ops:
        return fallback_ops, "ops_rule_fallback:selector_empty"

    reason = str(parsed.get("reason", "llm_ops_selector"))
    return chosen_ops, reason


async def stream_answer_with_llm(
    question: str,
    citations: list[Citation],
    analysis: AnalysisResult,
    timeout_s: float = 60.0,
) -> AsyncGenerator[str, None]:
    if not llm_enabled():
        return

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
        "analysis": _analysis_for_llm(analysis),
    }

    if langchain_enabled():
        try:
            chain = _answer_chain(timeout_s=timeout_s)
            payload = json.dumps(context, ensure_ascii=False)
            async for chunk in chain.astream({"payload": payload}):
                text = _extract_text_from_langchain_message(chunk)
                if text:
                    yield text
            return
        except Exception:  # noqa: BLE001
            pass

    try:
        async for token in _legacy_stream_answer_with_llm(context=context, timeout_s=timeout_s):
            yield token
    except Exception:  # noqa: BLE001
        return


def _router_chain(timeout_s: float):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT),
            ("human", "question={question}\nimage_available={image_available}"),
        ]
    )
    return prompt | _build_chat_model(timeout_s=timeout_s).with_structured_output(RouterOutput)


def _ops_selector_chain(timeout_s: float):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", OPS_SELECTOR_PROMPT),
            ("human", "question={question}\nallowed_ops={allowed_ops}"),
        ]
    )
    return prompt | _build_chat_model(timeout_s=timeout_s).with_structured_output(OpsSelectorOutput)


def _answer_chain(timeout_s: float):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{payload}"),
        ]
    )
    return prompt | _build_chat_model(timeout_s=timeout_s)


def _build_chat_model(timeout_s: float):
    return ChatOpenAI(  # type: ignore[operator]
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=timeout_s,
    )


def _normalize_selected_ops(raw_ops: Any) -> list[str]:
    if not isinstance(raw_ops, list):
        return []
    chosen_ops: list[str] = []
    for op in raw_ops:
        if not isinstance(op, str):
            continue
        normalized = op.strip()
        if normalized in SUPPORTED_OPS and normalized not in chosen_ops:
            chosen_ops.append(normalized)
        if len(chosen_ops) >= 3:
            break
    return chosen_ops


def _extract_text_from_langchain_message(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


async def _legacy_json_prompt_call(
    system_prompt: str,
    user_payload: dict,
    timeout_s: float,
) -> dict | None:
    response_text = await _legacy_text_prompt_call(
        system_prompt=system_prompt,
        user_payload=user_payload,
        timeout_s=timeout_s,
    )
    return _extract_json_object(response_text or "")


async def _legacy_text_prompt_call(
    system_prompt: str,
    user_payload: dict,
    timeout_s: float,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        return await _request_text_with_responses_or_chat(
            client=client,
            headers=headers,
            system_prompt=system_prompt,
            user_payload=user_payload,
        )


async def _legacy_stream_answer_with_llm(
    context: dict,
    timeout_s: float,
) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    chat_body = {
        "model": settings.llm_model,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    }
    chat_url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        async with client.stream("POST", chat_url, headers=headers, json=chat_body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if isinstance(delta, str) and delta:
                    yield delta


async def _request_text_with_responses_or_chat(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    system_prompt: str,
    user_payload: dict,
) -> str | None:
    responses_body = {
        "model": settings.llm_model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}],
            },
        ],
    }

    responses_url = f"{settings.llm_base_url.rstrip('/')}/responses"
    response = await client.post(responses_url, headers=headers, json=responses_body)
    if response.is_success:
        parsed = _parse_responses_output(response.json())
        if parsed:
            return parsed

    chat_body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    chat_url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    chat_response = await client.post(chat_url, headers=headers, json=chat_body)
    chat_response.raise_for_status()
    return _parse_chat_completions_output(chat_response.json())


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


def _analysis_for_llm(analysis: AnalysisResult) -> dict:
    return {
        "invoked": analysis.invoked,
        "error": analysis.error,
        "ops": [
            {
                "name": op.name,
                "summary": op.summary,
                "stats": op.stats,
            }
            for op in analysis.ops
        ],
    }


def _fallback_ops(question: str) -> list[str]:
    text = question.lower()
    if any(keyword in text for keyword in ("구름", "cloud")):
        return ["cloud_mask_like"]
    if any(keyword in text for keyword in ("식생", "녹지", "vegetation", "mask")):
        return ["masking_like"]
    if any(keyword in text for keyword in ("경계", "edge", "윤곽")):
        return ["edges"]
    if any(keyword in text for keyword in ("밝", "임계", "threshold")):
        return ["threshold"]
    if any(keyword in text for keyword in ("노이즈", "morphology", "morph")):
        return ["morphology"]
    return ["edges"]


def _extract_json_object(raw_text: str) -> dict | None:
    raw_text = raw_text.strip()
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = raw_text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None
