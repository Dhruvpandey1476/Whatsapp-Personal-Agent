"""Provider-agnostic LLM client for Gemini, Groq and OpenRouter.

All three expose an OpenAI-compatible `/chat/completions` endpoint with function calling, so one adapter
covers them. Switch providers with LLM_PROVIDER in .env; no code change.

The agent loop keeps history in a neutral block format ({"type": "text" | "tool_use" | "tool_result"}).
This module converts it to the OpenAI wire format on the way out and normalises the response on the way back.
"""

import asyncio
import json
import logging
import uuid
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "google/gemini-2.5-flash"},
}
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


def _clean_schema(schema: Any) -> Any:
    """Drop JSON-schema keys some providers reject (Gemini is strict about `default`)."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k not in ("default", "cache_control")}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"], "parameters": _clean_schema(t["input_schema"])}}
        for t in tools
    ]  # fmt: skip


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "\n".join(b["text"] for b in content if b["type"] == "text").strip()
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                for b in content if b["type"] == "tool_use"
            ]  # fmt: skip
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            # Tool results must directly follow the assistant tool_calls message, then any user text.
            for b in content:
                if b["type"] == "tool_result":
                    body = b["content"] if not b.get("is_error") else f"ERROR: {b['content']}"
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
            text = "\n".join(b["text"] for b in content if b["type"] == "text").strip()
            if text:
                out.append({"role": "user", "content": text})
    return out


def from_openai_response(data: dict) -> dict:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response: {str(data)[:300]}") from exc
    blocks: list[dict] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append({"type": "tool_use", "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                       "name": fn.get("name", ""), "input": args if isinstance(args, dict) else {}})  # fmt: skip
    # Some providers report finish_reason "stop" even when they returned tool calls.
    has_calls = any(b["type"] == "tool_use" for b in blocks)
    return {"content": blocks, "stop_reason": "tool_use" if has_calls else "end_turn"}


class OpenAICompatibleLLM:
    def __init__(self, provider: str | None = None, http: httpx.AsyncClient | None = None) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}")
        preset = PROVIDERS[self.provider]
        self.base_url = (settings.llm_base_url or preset["base_url"]).rstrip("/")
        self.model = settings.llm_model or preset["model"]
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        if self.provider == "openrouter":
            headers["X-Title"] = settings.app_name
        self.http = http or httpx.AsyncClient(timeout=60, headers=headers)

    async def create(self, *, system: str, tools: list[dict], messages: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "tools": to_openai_tools(tools),
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        for attempt in range(3):
            try:
                resp = await self.http.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise LLMError(f"{self.provider} unreachable: {exc}") from exc
            else:
                if resp.status_code == 200:
                    return from_openai_response(resp.json())
                if resp.status_code not in RETRYABLE or attempt == 2:
                    raise LLMError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:300]}")
                log.warning("%s HTTP %s, retrying", self.provider, resp.status_code)
            await asyncio.sleep(1.5 * 2**attempt)
        raise LLMError("unreachable")
