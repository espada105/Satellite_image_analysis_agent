from fastapi import HTTPException

from app.adapter.client.http import get_json, post_json


async def forward_post(url: str, payload: dict, headers: dict[str, str]) -> dict:
    try:
        return await post_json(url, payload, headers)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"upstream post failed: {exc}") from exc


async def forward_get(url: str) -> dict:
    try:
        return await get_json(url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"upstream get failed: {exc}") from exc
