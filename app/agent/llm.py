"""Provider-agnostic LLM client for Gemini, Groq and OpenRouter.

All three expose an OpenAI-compatible `/chat/completions` endpoint with function calling, so one adapter
covers them. Switch providers with LLM_PROVIDER in .env; no code change.

The agent loop keeps history in a neutral block format ({"type": "text" | "tool_use" | "tool_result"}).
This module converts it to the OpenAI wire format on the way out and normalises the response on the way back.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3-flash-preview"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "qwen/qwen3.8-27b"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "google/gemini-2.5-flash"},
}
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Unavailable(LLMError):
    """The model can't answer right now (timeouts, 5xx, rate limit) even after retries: worth trying the fallback."""


class RateLimited(Unavailable):
    """Quota / rate limit exhausted even after retries."""


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


# Gemini 3 rejects earlier tool calls without a thought signature. Calls made by ANOTHER model (e.g. Groq, before
# a fallback) have none; Google documents this placeholder for conversations transferred from other models.
GEMINI_PLACEHOLDER_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def to_openai_messages(system: str, messages: list[dict], provider: str = "") -> list[dict]:
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
                 "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)},
                 # Gemini 3 requires its thought signature to be echoed back with the tool call
                 **({"extra_content": b["extra"]} if b.get("extra")
                    else {"extra_content": GEMINI_PLACEHOLDER_SIGNATURE} if provider == "gemini" else {})}
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
    # Reasoning models (Qwen 3, DeepSeek R1...) may inline their hidden reasoning; never show it to the user.
    content = re.sub(r"(?s)<think>.*?(</think>|$)", "", msg.get("content") or "").strip()
    if content:
        blocks.append({"type": "text", "text": content})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append({"type": "tool_use", "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                       "name": fn.get("name", ""), "input": args if isinstance(args, dict) else {},
                       **({"extra": call["extra_content"]} if call.get("extra_content") else {})})  # fmt: skip
    # Some providers report finish_reason "stop" even when they returned tool calls.
    has_calls = any(b["type"] == "tool_use" for b in blocks)
    return {"content": blocks, "stop_reason": "tool_use" if has_calls else "end_turn"}


# "tokens per day", "requests per day", "daily limit", "quota": waiting a few seconds won't help.
QUOTA_EXHAUSTED = re.compile(r"per day|\bTPD\b|\bRPD\b|daily|quota", re.I)
QUOTA_COOLDOWN_S = 600


def key_for(provider: str) -> str:
    """API key for a provider: its own setting (GROQ_/GEMINI_/OPENROUTER_API_KEY), else the main/fallback key."""
    explicit = {"groq": settings.groq_api_key, "gemini": settings.gemini_api_key,
                "openrouter": settings.openrouter_api_key}.get(provider, "")  # fmt: skip
    if explicit:
        return explicit
    if provider == settings.llm_provider:
        return settings.llm_api_key
    if provider == settings.llm_fallback_provider.lower():
        return settings.llm_fallback_api_key
    return ""


def fallback_chain(main_provider: str, main_model: str) -> list[tuple[str, str]]:
    """[(provider, model)] tried in order after the main model. LLM_FALLBACKS ("openrouter:x:free,gemini:y") wins;
    otherwise LLM_FALLBACK_PROVIDER + LLM_FALLBACK_MODEL (a comma list) are used. Providers without a key are skipped."""
    chain: list[tuple[str, str]] = []
    if settings.llm_fallbacks.strip():
        for item in settings.llm_fallbacks.split(","):
            provider, _, model = item.strip().partition(":")  # model ids may contain ':' (":free")
            if provider.lower() in PROVIDERS:
                chain.append((provider.lower(), model.strip()))
    else:
        provider = settings.llm_fallback_provider.lower() or main_provider
        models = [m.strip() for m in settings.llm_fallback_model.split(",") if m.strip()]
        chain = [(provider, m) for m in models] or ([(provider, "")] if provider != main_provider else [])
    return [(p, m) for p, m in chain if (p, m) != (main_provider, main_model) and (p == main_provider or key_for(p))]


class OpenAICompatibleLLM:
    def __init__(self, provider: str | None = None, http: httpx.AsyncClient | None = None, *,
                 api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:  # fmt: skip
        self.provider = (provider or settings.llm_provider).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}")
        preset = PROVIDERS[self.provider]
        main = provider is None or self.provider == settings.llm_provider
        self.base_url = (base_url or (settings.llm_base_url if main else "") or preset["base_url"]).rstrip("/")
        self.model = model or (settings.llm_model if main else "") or preset["model"]
        headers = {"Authorization": f"Bearer {api_key or key_for(self.provider) or settings.llm_api_key}"}
        if self.provider == "openrouter":
            headers["X-Title"] = settings.app_name
        self.http = http or httpx.AsyncClient(timeout=25, headers=headers)
        self._clients: dict[str, OpenAICompatibleLLM] = {}
        self._cooldown: dict[tuple[str, str], float] = {}  # (provider, model) -> skip until (monotonic time)

    def _client_for(self, provider: str) -> "OpenAICompatibleLLM":
        if provider == self.provider:
            return self
        if provider not in self._clients:
            self._clients[provider] = OpenAICompatibleLLM(provider, api_key=key_for(provider))
        return self._clients[provider]

    async def create(self, *, system: str, tools: list[dict], messages: list[dict]) -> dict:
        """Main model first, then the fallback chain. Falls through on anything except the main model's bad
        credentials: outages, rate limits, timeouts, and 400s such as Groq's tool_use_failed. Models whose daily
        quota is used up are skipped for a while instead of being retried on every message."""
        attempts = [(self, self.model), *[(self._client_for(p), m) for p, m in fallback_chain(self.provider, self.model)]]
        error: LLMError | None = None
        now = time.monotonic()
        for client, model in attempts:
            model = model or client.model
            if self._cooldown.get((client.provider, model), 0) > now:
                continue
            if error is not None:
                log.warning("falling back to %s:%s (%s)", client.provider, model, str(error)[:160])
            try:
                return await client._create(model, system, tools, messages)
            except LLMError as exc:
                if client is self and exc.status in (401, 403):
                    raise  # misconfigured main key: fail loudly instead of hiding it
                if isinstance(exc, RateLimited) and QUOTA_EXHAUSTED.search(str(exc)):
                    self._cooldown[(client.provider, model)] = now + QUOTA_COOLDOWN_S
                    log.warning("%s:%s is out of quota; skipping it for %d min", client.provider, model, QUOTA_COOLDOWN_S // 60)
                error = exc
        raise error or Unavailable("all models are cooling down after hitting their quotas")

    async def _create(self, model: str, system: str, tools: list[dict], messages: list[dict]) -> dict:
        payload = {
            "model": model,
            "messages": to_openai_messages(system, messages, self.provider),
            "tools": to_openai_tools(tools),
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": settings.llm_max_tokens,
        }
        if self.provider == "groq" and "qwen" in model:
            payload["reasoning_effort"] = "none"  # no hidden thinking: faster, and far fewer tokens per minute
        for attempt in range(3):
            try:
                resp = await self.http.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                # A hung model rarely recovers on the same model: retry once, then let create() fall back.
                if attempt >= 1:
                    raise Unavailable(f"{self.provider} unreachable: {exc.__class__.__name__} {exc}") from exc
                log.warning("%s %s, retrying", self.provider, exc.__class__.__name__)
            else:
                if resp.status_code == 200:
                    return from_openai_response(resp.json())
                if resp.status_code == 429 and QUOTA_EXHAUSTED.search(resp.text):
                    # a daily quota doesn't come back in seconds: don't retry, let the chain move on
                    raise RateLimited(f"{self.provider} HTTP 429: {resp.text[:300]}", 429)
                if resp.status_code not in RETRYABLE or attempt == 2:
                    err = (RateLimited if resp.status_code == 429
                           else Unavailable if resp.status_code in RETRYABLE else LLMError)  # fmt: skip
                    raise err(f"{self.provider} HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
                log.warning("%s HTTP %s, retrying", self.provider, resp.status_code)
            await asyncio.sleep(1.5 * 2**attempt)
        raise LLMError("unreachable")
