"""The agent loop: Claude + tools, with per-user memory.

    user text -> [LLM -> tool calls -> tool results]* -> final text (+ queued WhatsApp lists/buttons)

The LLM client sits behind a tiny protocol so tests can use a scripted fake (no network, deterministic).
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Protocol

from app.agent.tools import REGISTRY, TOOLS, ToolContext, ToolError
from app.config import settings
from app.models import ToolTrace

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are {name}'s personal assistant on WhatsApp. You get things done for them: \
flight bookings, train (IRCTC-style) bookings and food delivery orders. Today is {today} ({weekday}), \
time {time} IST.

How to work:
- Act, don't interrogate. Use sensible defaults (1 adult, cheapest, today for food) and the saved \
profile; ask only for what is truly missing, all in one short question.
- Resolve relative dates ("next Friday", "kal") yourself into YYYY-MM-DD.
- Search results are automatically shown to the user as a tappable list; summarise only the top 2-3 \
in one line each and recommend one.
- When the user picks an option, call prepare_booking. It sends Confirm/Cancel buttons. Never claim \
something is booked, ordered or paid until a tool result says it is confirmed. Payment is always done \
by the user on the payment link; you never pay or confirm on their behalf, even if they say "yes".
- When the user shares personal details (name, age, gender, address, preferences), save them with \
update_profile without asking.
- Only cancel when the user explicitly asks to cancel a specific booking.
- You can only do flights, trains and food. For anything else say briefly what you can do instead.
- Style: WhatsApp-friendly, short, warm, emojis sparingly, *bold* for key facts. Reply in the user's \
language (English / Hindi / Hinglish).

Saved profile (long-term memory): {profile}"""


class LLM(Protocol):
    async def create(self, *, system: str, tools: list[dict], messages: list[dict]) -> dict:
        """Return {"content": [blocks as dicts], "stop_reason": str}."""
        ...


class AnthropicLLM:
    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=2, timeout=60)
        # Cache the (static) tool definitions: cheaper and faster on every turn.
        self.tools = [*TOOLS[:-1], {**TOOLS[-1], "cache_control": {"type": "ephemeral"}}]

    async def create(self, *, system: str, tools: list[dict], messages: list[dict]) -> dict:
        resp = await self.client.messages.create(
            model=settings.llm_model, max_tokens=1024, system=system, tools=self.tools, messages=messages
        )
        blocks: list[dict] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return {"content": blocks, "stop_reason": resp.stop_reason}


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = AnthropicLLM()
    return _llm


def set_llm(llm: LLM | None) -> None:
    global _llm
    _llm = llm


def _system(ctx: ToolContext) -> str:
    now = datetime.now(settings.tz)
    profile = ctx.contact.profile or {}
    return SYSTEM_PROMPT.format(
        name=profile.get("full_name") or ctx.contact.name or "the user",
        today=now.date().isoformat(),
        weekday=now.strftime("%A"),
        time=now.strftime("%H:%M"),
        profile=json.dumps(profile, ensure_ascii=False) if profile else "(empty: nothing saved yet)",
    )


def _trim(history: list[dict], limit: int) -> list[dict]:
    """Keep the last `limit` messages, starting at a plain user turn so no tool_result is orphaned."""
    h = history[-limit:]
    def plain_user(m: dict) -> bool:
        c = m["content"]
        return m["role"] == "user" and (isinstance(c, str) or all(b.get("type") == "text" for b in c))

    while h and not plain_user(h[0]):
        h = h[1:]
    return h


def _merge_roles(history: list[dict]) -> list[dict]:
    """Merge consecutive same-role turns (e.g. a system note followed by the user's message)."""
    out: list[dict] = []
    for m in history:
        if out and out[-1]["role"] == m["role"]:
            prev = out[-1]["content"]
            as_blocks = lambda c: [{"type": "text", "text": c}] if isinstance(c, str) else c  # noqa: E731
            out[-1] = {"role": m["role"], "content": as_blocks(prev) + as_blocks(m["content"])}
        else:
            out.append(m)
    return out


async def _run_tool(ctx: ToolContext, name: str, args: dict) -> tuple[Any, bool]:
    fn = REGISTRY.get(name)
    started = time.perf_counter()
    try:
        if fn is None:
            raise ToolError(f"Unknown tool {name}")
        result, is_error = await fn(ctx, **args), False
    except (ToolError, ValueError, TypeError) as exc:  # bad args / business rule: let the model recover
        result, is_error = {"error": str(exc)}, True
    except Exception:
        log.exception("tool %s crashed", name)
        result, is_error = {"error": "Internal error in tool; apologise and try another way."}, True
    ctx.db.add(ToolTrace(contact_id=ctx.contact.id, tool=name, input=args, output=result, is_error=is_error,
                         ms=int((time.perf_counter() - started) * 1000)))  # fmt: skip
    return result, is_error


async def run_agent(ctx: ToolContext, user_text: str) -> str:
    """Run one user turn to completion. Returns the text to send; interactive messages land in ctx.outbox."""
    llm = get_llm()
    history = _trim([*(ctx.contact.history or []), {"role": "user", "content": user_text}], settings.history_messages)
    messages = _merge_roles(history)
    reply = ""

    for _ in range(settings.agent_max_steps):
        resp = await llm.create(system=_system(ctx), tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp["content"]})
        texts = [b["text"] for b in resp["content"] if b["type"] == "text" and b["text"].strip()]
        if texts:
            reply = "\n\n".join(texts)
        calls = [b for b in resp["content"] if b["type"] == "tool_use"]
        if resp["stop_reason"] != "tool_use" or not calls:
            break
        results = []
        for call in calls:
            out, is_error = await _run_tool(ctx, call["name"], call["input"] or {})
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": json.dumps(out, ensure_ascii=False, default=str), "is_error": is_error})  # fmt: skip
        messages.append({"role": "user", "content": results})
    else:
        reply = reply or "Sorry, that took too many steps. Could you rephrase or break it into smaller requests?"

    ctx.contact.history = _trim(messages, settings.history_messages)
    await ctx.db.commit()
    return reply


async def remember(ctx: ToolContext, note: str) -> None:
    """Record an out-of-band event (button tap, payment) so the agent knows about it next turn."""
    hist = list(ctx.contact.history or [])
    hist.append({"role": "user", "content": f"[system note: {note}]"})
    ctx.contact.history = hist
    await ctx.db.commit()
