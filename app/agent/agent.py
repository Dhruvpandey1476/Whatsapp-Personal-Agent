"""The agent loop: LLM (Gemini / Groq / OpenRouter) + tools, with per-user memory.

    user text -> [LLM -> tool calls -> tool results]* -> final text (+ queued WhatsApp lists/buttons)

The LLM client sits behind a tiny protocol so tests can use a scripted fake (no network, deterministic).
"""

import json
import re
import logging
import time
from datetime import datetime
from typing import Any, Protocol

from app.agent.llm import OpenAICompatibleLLM
from app.agent.tools import REGISTRY, TOOLS, ToolContext, ToolError
from app.config import settings
from app.models import ToolTrace

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are {name}'s personal AI assistant on WhatsApp: smart, friendly and genuinely \
useful, like a capable friend. Today is {today} ({weekday}), time {time} IST.

What you can do:
1. *Get things done* (actions): book flights (domestic + international), book trains in India, order \
food (pay online or cash on delivery), show / cancel bookings.
2. *Find live information*: web_search for anything current or factual (news, movies, sports, prices, \
places, recommendations, how-tos), read_webpage for details, get_weather for any city.
3. *Just talk*: answer general questions, explain things, brainstorm, plan trips, give advice, small talk.

How to work:
- Be helpful on ANY topic. Answer from your own knowledge for stable facts; use web_search when the answer \
depends on recent events or you aren't sure. Never invent current facts (scores, releases, prices, news). \
Mention the source site briefly when you used search.
- Give real answers, not homework. When asked for a recommendation or opinion (movie, series, book, food, \
place, gadget), confidently name 2-3 specific picks with a one-line reason each, using your own knowledge \
plus any search results; then ask a quick follow-up (genre, platform) if useful. Never say you can't give \
recommendations, and never ask permission before searching; just do it.
- Search results are raw material: read them and answer the question in your own words. Never reply with only \
a list of links or article titles. If results are just headlines, combine them with what you know; use \
read_webpage on a promising (non news.google.com) result when you need details.
- Links: share at most one or two, only when the user needs them to act (book, buy, read more), as a plain \
URL. Never Markdown [text](url) links, and never news.google.com links.
- For tasks you can't perform yourself (hotels, cabs, movie tickets, shopping...), still help: search for \
options, compare them, recommend one, and share a link so the user can finish it.
- Connect things proactively: e.g. after a flight booking, offer the destination weather; when recommending \
movies for tonight, offer to order food.
- Act, don't interrogate. Use sensible defaults (1 adult, cheapest, nearest date) and the saved profile; ask \
only for what's truly missing, in one short question. Resolve relative dates ("next Friday", "kal") yourself.
- Call each search tool at most once per user message unless the first result was clearly insufficient.
- Booking search results are shown to the user as a tappable list automatically; summarise only the top \
2-3 in one line each and recommend one.
- When the user picks an option, call prepare_booking. It sends Confirm/Cancel buttons. Never claim something \
is booked, ordered or paid until a tool result or system note says so. Payment is always done by the user \
(payment link, or cash on delivery for food); you never pay or confirm on their behalf, even if they say "yes".
- When the user shares personal details (name, age, gender, address, preferences), save them with \
update_profile without asking. Only cancel a booking when explicitly asked.
- Flight, train and restaurant inventory is a demo dataset (not live airline/IRCTC/restaurant systems). If \
asked whether those prices are real-time, say so honestly. Web search and weather are live.
- Style: WhatsApp-friendly, concise, warm, emojis sparingly. WhatsApp formatting only: *bold* with single \
asterisks, _italic_; no markdown headings, tables or **double asterisks**.
- Language: always reply in the language AND script of the user's LATEST message, never of earlier messages. \
English → English. Hinglish (Hindi written in English letters, e.g. "kal sasti flight") → Hinglish in English \
letters. Use Devanagari (हिंदी) only if the user wrote in Devanagari.

Saved profile (long-term memory): {profile}"""

HINGLISH_WORDS = {
    "kal", "aaj", "parso", "sasti", "sasta", "sabse", "dikhao", "batao", "chahiye", "karo", "kar", "karna", "kya",
    "hai", "hain", "mujhe", "mera", "meri", "ke", "ki", "ko", "se", "aur", "nahi", "haan", "acha", "accha", "bhai",
    "kitna", "kitne", "kaise", "kab", "kahan", "wala", "wali", "doon", "dena", "lena", "jana", "hoga", "bhej",
}  # fmt: skip


def reply_language(text: str) -> str:
    """Decide the reply language/script from the user's latest message (deterministic, so the model can't drift)."""
    if re.search(r"[ऀ-ॿ]", text):
        return "Hindi in Devanagari script"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in HINGLISH_WORDS for w in words)
    if words and (hits >= 2 or hits / len(words) >= 0.25):
        return "Hinglish (Hindi in English/Latin letters, never Devanagari)"
    return "English"


class LLM(Protocol):
    async def create(self, *, system: str, tools: list[dict], messages: list[dict]) -> dict:
        """Return {"content": [blocks as dicts], "stop_reason": str}."""
        ...


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = OpenAICompatibleLLM()
    return _llm


def set_llm(llm: LLM | None) -> None:
    global _llm
    _llm = llm


def active_tools() -> tuple[list[dict], dict]:
    """Tool schemas + implementations for this deployment (browser tools only when enabled)."""
    if not settings.browser_enabled:
        return TOOLS, REGISTRY
    from app.agent.browser_tools import BROWSER_REGISTRY, BROWSER_TOOLS

    return [*TOOLS, *BROWSER_TOOLS], {**REGISTRY, **BROWSER_REGISTRY}


def _system(ctx: ToolContext, user_text: str = "") -> str:
    now = datetime.now(settings.tz)
    profile = ctx.contact.profile or {}
    prompt = SYSTEM_PROMPT.format(
        name=profile.get("full_name") or ctx.contact.name or "the user",
        today=now.date().isoformat(),
        weekday=now.strftime("%A"),
        time=now.strftime("%H:%M"),
        profile=json.dumps(profile, ensure_ascii=False) if profile else "(empty: nothing saved yet)",
    )
    if settings.browser_enabled:
        from app.agent.browser_tools import browser_prompt

        prompt += browser_prompt().replace("{wa_id}", ctx.contact.wa_id or "unknown")
    else:
        prompt += (
            "\n\nNo browser on this server: you can't log in to or click around websites here (e.g. SkyTrip). If the "
            "user asks you to log in to or operate a website, say clearly that the browser agent runs only in the "
            "local demo mode, and offer what you can do instead (e.g. the normal flight tools). If they ask you to "
            "read a page or look something up on Wikipedia, use read_webpage on the article URL "
            "(e.g. https://en.wikipedia.org/wiki/Third_Battle_of_Panipat) and answer from it."
        )
    if user_text:
        prompt += f"\n\nThe user's latest message is in {reply_language(user_text)}. Write your whole reply in that."
    return prompt


def _trim(history: list[dict], limit: int) -> list[dict]:
    """Keep the last `limit` messages, starting at a plain user turn so no tool_result is orphaned."""
    h = history[-limit:]
    def plain_user(m: dict) -> bool:
        c = m["content"]
        return m["role"] == "user" and (isinstance(c, str) or all(b.get("type") == "text" for b in c))

    while h and not plain_user(h[0]):
        h = h[1:]
    return h


# "I'll check / let me search / give me a second" said without actually calling a tool.
EMPTY_PROMISE = re.compile(r"\b(let me|i'?ll|i will|i am going to|i'm going to|give me (just )?a (sec|second|moment)|"
                           r"one (sec|second|moment)|hold on|bas ek second|abhi (check|dekh))\b", re.I)  # fmt: skip

OLD_RESULT_CHARS = 300  # tool results from earlier turns are kept only as a short summary


def _compact(history: list[dict], keep_current: bool = True) -> list[dict]:
    """Shrink tool results from earlier turns (search results are big), keeping every request well under Groq's
    free-tier input-tokens-per-minute limit. keep_current=True leaves the current turn's results intact (needed to
    answer now); False shrinks all of them (for storage, since they're 'earlier' by the next message)."""
    last_user = len(history) if not keep_current else max(
        (i for i, m in enumerate(history) if m["role"] == "user" and isinstance(m["content"], str)),
        default=len(history))  # fmt: skip
    out = []
    for i, m in enumerate(history):
        if i < last_user and m["role"] == "user" and isinstance(m["content"], list):
            m = {**m, "content": [
                {**b, "content": b["content"][:OLD_RESULT_CHARS] + "…(older result, shortened)"}
                if b.get("type") == "tool_result" and len(b.get("content", "")) > OLD_RESULT_CHARS else b
                for b in m["content"]]}  # fmt: skip
        out.append(m)
    return out


def _with_language_hint(messages: list[dict], language: str) -> list[dict]:
    """Attach the reply language to the user's latest real message (a system-prompt instruction loses to a history
    full of replies in another language). It must go on the user's own text, NOT after tool results: there it
    reaches the model as a separate user message, and the model answers that instead ("Got it! How can I help?")."""
    hint = f"(Reply in {language}; ignore the language of earlier replies.)"
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] != "user":
            continue
        content = m["content"]
        if isinstance(content, str):
            new = f"{content}\n\n{hint}"
        elif any(b.get("type") == "tool_result" for b in content):
            continue  # tool results, not something the user said
        else:
            new = [*content, {"type": "text", "text": hint}]
        return [*messages[:i], {**m, "content": new}, *messages[i + 1:]]
    return messages


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
    fn = active_tools()[1].get(name)
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
    await ctx.db.commit()  # keep the trace even if a later step of this turn fails (that's when you need it most)
    return result, is_error


async def run_agent(ctx: ToolContext, user_text: str) -> str:
    """Run one user turn to completion. Returns the text to send; interactive messages land in ctx.outbox."""
    llm = get_llm()
    history = _trim([*(ctx.contact.history or []), {"role": "user", "content": user_text}], settings.history_messages)
    messages = _merge_roles(_compact(history))
    reply = ""

    # Language is judged on what the user actually said (not on our appended voice-note instruction).
    said = user_text.split("\n\n[voice note")[0]
    system, language = _system(ctx, said), reply_language(said)
    if settings.browser_enabled:
        from app.browser.manager import get_browser

        try:
            state = await get_browser().describe(ctx.contact.wa_id)
            system += f"\n\nAt the start of this message, the user's browser showed: {state}. Trust this over the chat history."
        except Exception:  # noqa: BLE001 - a hint, never a reason to fail the turn
            log.warning("could not read browser state for %s", ctx.contact.wa_id)
    tools = active_tools()[0]
    nudged = False
    for _ in range(settings.agent_max_steps):
        # The hint is added only to what's sent, never stored in the conversation history.
        resp = await llm.create(system=system, tools=tools, messages=_with_language_hint(messages, language))
        messages.append({"role": "assistant", "content": resp["content"]})
        texts = [b["text"] for b in resp["content"] if b["type"] == "text" and b["text"].strip()]
        if texts:
            reply = "\n\n".join(texts)
        calls = [b for b in resp["content"] if b["type"] == "tool_use"]
        if resp["stop_reason"] != "tool_use" or not calls:
            if texts and not nudged and EMPTY_PROMISE.search(reply) and not reply.rstrip().endswith("?"):
                # "Let me check… give me a second" with no tool call: the user would wait forever. Send it back once.
                nudged = True
                messages.append({"role": "user", "content": "[system note: you said you'd do it but didn't call any "
                                 "tool. Do it now with the tools, or ask the user one clear question.]"})  # fmt: skip
                continue
            break
        results = []
        for call in calls:
            out, is_error = await _run_tool(ctx, call["name"], call["input"] or {})
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": json.dumps(out, ensure_ascii=False, default=str), "is_error": is_error})  # fmt: skip
        messages.append({"role": "user", "content": results})
    else:
        reply = reply or "Sorry, that took too many steps. Could you rephrase or break it into smaller requests?"

    ctx.contact.history = _trim(_compact(messages, keep_current=False), settings.history_messages)
    await ctx.db.commit()
    return reply


async def remember(ctx: ToolContext, note: str) -> None:
    """Record an out-of-band event (button tap, payment) so the agent knows about it next turn."""
    hist = list(ctx.contact.history or [])
    hist.append({"role": "user", "content": f"[system note: {note}]"})
    ctx.contact.history = hist
    await ctx.db.commit()
