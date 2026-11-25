"""The OpenAI-compatible adapter (Gemini / Groq / OpenRouter), tested against a fake HTTP transport."""

import asyncio
import json

import httpx
import pytest

from app.agent.llm import PROVIDERS, LLMError, OpenAICompatibleLLM, to_openai_messages, to_openai_tools
from app.agent.tools import TOOLS


def test_tools_schema_is_cleaned():
    tools = to_openai_tools(TOOLS)
    assert {t["function"]["name"] for t in tools} >= {"search_flights", "prepare_booking"}
    assert '"default"' not in json.dumps(tools)  # Gemini rejects `default`


def test_history_conversion_orders_tool_results_correctly():
    msgs = to_openai_messages("SYS", [
        {"role": "user", "content": "flight to goa"},
        {"role": "assistant", "content": [{"type": "text", "text": "Searching"},
                                          {"type": "tool_use", "id": "c1", "name": "search_flights", "input": {"a": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}", "is_error": True},
                                     {"type": "text", "text": "[system note: x]"}]},
    ])  # fmt: skip
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "search_flights", "arguments": '{"a": 1}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: {}"}


@pytest.mark.parametrize("provider", list(PROVIDERS))
async def test_each_provider_round_trip(provider):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        # Gemini-style quirk: finish_reason "stop" with tool calls and no id
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "get_menu", "arguments": '{"restaurant_id":"R1"}'}}],
        }}]})  # fmt: skip

    llm = OpenAICompatibleLLM(provider, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = await llm.create(system="S", tools=TOOLS, messages=[{"role": "user", "content": "menu"}])
    assert seen["url"] == PROVIDERS[provider]["base_url"] + "/chat/completions"
    assert seen["body"]["model"] == PROVIDERS[provider]["model"]
    assert out["stop_reason"] == "tool_use"
    [call] = out["content"]
    assert call["name"] == "get_menu" and call["input"] == {"restaurant_id": "R1"} and call["id"]


async def test_non_retryable_error_raises():
    llm = OpenAICompatibleLLM("groq", http=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"error": "bad key"}))))  # fmt: skip
    with pytest.raises(LLMError, match="401"):
        await llm.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        OpenAICompatibleLLM("anthropic")


async def test_rate_limit_falls_back_to_second_model(monkeypatch):
    from app.agent import llm as llm_mod

    monkeypatch.setattr(llm_mod.settings, "llm_fallback_model", "small-model")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm_mod.asyncio, "sleep", lambda s: real_sleep(0))  # skip backoff waits
    models = []

    def handler(req):
        model = json.loads(req.content)["model"]
        models.append(model)
        if model != "small-model":
            return httpx.Response(429, json={"error": "quota"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})

    llm = OpenAICompatibleLLM("gemini", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = await llm.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])
    assert out["content"][0]["text"] == "hi" and models[-1] == "small-model" and models.count("small-model") == 1


async def test_timeout_falls_back_to_second_model(monkeypatch):
    from app.agent import llm as llm_mod

    monkeypatch.setattr(llm_mod.settings, "llm_fallback_model", "small-model")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm_mod.asyncio, "sleep", lambda s: real_sleep(0))
    calls = []

    def handler(req):
        model = json.loads(req.content)["model"]
        calls.append(model)
        if model != "small-model":
            raise httpx.ReadTimeout("hung", request=req)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

    llm = OpenAICompatibleLLM("gemini", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = await llm.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])
    assert out["content"][0]["text"] == "ok"
    assert calls.count("small-model") == 1 and len(calls) == 3  # 2 tries on the hung model, then fallback


async def test_fallback_to_a_different_provider(monkeypatch):
    from app.agent import llm as llm_mod

    monkeypatch.setattr(llm_mod.settings, "llm_fallback_provider", "gemini")
    monkeypatch.setattr(llm_mod.settings, "llm_fallback_model", "gemini-flash-lite-latest")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm_mod.asyncio, "sleep", lambda s: real_sleep(0))
    urls = []

    def handler(req):
        urls.append(str(req.url))
        if "groq" in str(req.url):
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "from gemini"}}]})

    transport = httpx.MockTransport(handler)
    main = OpenAICompatibleLLM("groq", http=httpx.AsyncClient(transport=transport))
    main._fallback = OpenAICompatibleLLM("gemini", http=httpx.AsyncClient(transport=transport),
                                         model="gemini-flash-lite-latest")  # fmt: skip
    out = await main.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])
    assert out["content"][0]["text"] == "from gemini"
    assert "groq" in urls[0] and "generativelanguage" in urls[-1]


def test_reasoning_is_never_shown():
    from app.agent.llm import from_openai_response

    out = from_openai_response({"choices": [{"message": {"content": "<think>user wants Goa</think>Here you go!"}}]})
    assert out["content"] == [{"type": "text", "text": "Here you go!"}]
    assert from_openai_response({"choices": [{"message": {"content": "<think>unfinished"}}]})["content"] == []


def test_gemini_gets_placeholder_signature_for_tool_calls_from_another_model():
    history = [{"role": "user", "content": "flights"},
               {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "search_flights", "input": {}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}"}]}]  # fmt: skip
    call = to_openai_messages("S", history, "gemini")[2]["tool_calls"][0]
    assert call["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    assert "extra_content" not in to_openai_messages("S", history, "groq")[2]["tool_calls"][0]
    own = [{**history[1], "content": [{**history[1]["content"][0], "extra": {"google": {"thought_signature": "REAL"}}}]}]
    assert to_openai_messages("S", own, "gemini")[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "REAL"


async def test_groq_qwen_disables_reasoning_and_caps_tokens():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatibleLLM("groq", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)), model="qwen/qwen3.8-27b")
    await llm.create(system="S", tools=TOOLS, messages=[{"role": "user", "content": "hi"}])
    assert seen["reasoning_effort"] == "none" and seen["max_tokens"] <= 1000


async def test_groq_tool_use_failed_400_falls_back_but_bad_key_does_not(monkeypatch):
    from app.agent import llm as llm_mod

    monkeypatch.setattr(llm_mod.settings, "llm_fallback_provider", "gemini")
    status = {"code": 400}

    def handler(req):
        if "groq" in str(req.url):
            return httpx.Response(status["code"], json={"error": {"code": "tool_use_failed", "message": "Failed to call a function"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "from gemini"}}]})

    t = httpx.MockTransport(handler)
    main = OpenAICompatibleLLM("groq", http=httpx.AsyncClient(transport=t))
    main._fallback = OpenAICompatibleLLM("gemini", http=httpx.AsyncClient(transport=t), model="g")
    out = await main.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])
    assert out["content"][0]["text"] == "from gemini"
    status["code"] = 401
    with pytest.raises(LLMError, match="401"):
        await main.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])
