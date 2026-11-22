# WhatsApp Personal AI Agent

A personal assistant that lives in WhatsApp and **gets tasks done** (book flights, book trains, order food) and **talks about anything else**, using live
web search and weather. All by chatting in plain English, Hindi or Hinglish. It's built on the official **Meta WhatsApp Cloud API**,
with a **pluggable LLM (Gemini, Groq or OpenRouter)** as a tool-using agent and a **human-in-the-loop checkout**, so the AI can never spend money by itself.

```
You:    cheapest flight delhi to goa next friday evening
Agent:  Found 5 ✈️ Cheapest is *IndiGo 6E-412, 18:05 → 20:35, ₹4,280* (non-stop). Tap to pick 👇
        [View flights ▾]                          ← WhatsApp list message
You:    (taps IndiGo ₹4,280)
Agent:  I just need the passenger's full name, age and gender.
You:    Dhruv Pandey, 24, male                    ← saved to profile, never asked again
Agent:  ✈️ IndiGo 6E-412 DEL→GOI … 👤 Dhruv Pandey (24M)  *Total ₹4,280*
        [✅ Confirm & pay] [❌ Cancel]              ← reply buttons, handled in code, not by the LLM
You:    (taps Confirm)
Agent:  🔒 Tap below to pay securely  [Pay ₹4,280]  ← CTA URL button → checkout page
        (user pays on the page)
Agent:  🎉 Payment received: ₹4,280 · PNR: *QKZTRM*
```

## What it does

| Service | Capabilities |
|---|---|
| ✈️ **Flights** | Domestic + **international** (Dubai, Singapore, London, New York, Chicago, Toronto, Tokyo, Sydney…). Search by city or IATA code, with date, adults, sort (cheapest / earliest / fastest) and a departure time window. Overnight arrivals marked `(+1)` |
| 🚆 **Trains** | IRCTC-style search with per-class fares (1A/2A/3A/CC/SL) and live-looking availability or waitlist. Waitlisted classes can't be booked. 10-digit PNR |
| 🍔 **Food** | Find restaurants by dish or cuisine, veg-only filter, menus, multi-item cart, delivery fee and taxes, saved address. **Pay online or cash on delivery** |
| 🌐 **Live info** | `web_search` (Tavily if a key is set, otherwise DuckDuckGo → Google News → Wikipedia), `read_webpage` (public URLs only), `get_weather` (Open-Meteo, any city) |
| 💬 **General chat** | Answers any question and gives advice and recommendations. For things it can't book (hotels, cabs, movie tickets) it searches and shares links |
| 🧠 **Memory** | Long-term **profile** (name, age, gender, address, preferences) saved automatically, so "book the same for me" works. Short-term chat history per user |
| 📋 **Manage** | "show my bookings", "cancel FL-7K2Q9A" (refund shown). Paid food orders can't be cancelled |

## Architecture

![System architecture](docs/architecture.svg)

_Numbered steps follow one request from the user's message to the paid booking. The orange path is the money path: plain code plus explicit user actions, never the LLM._

```mermaid
flowchart LR
    U[User on WhatsApp] <--> M[Meta Cloud API]
    M -- signed webhook --> W[FastAPI /webhook<br/>HMAC verify · fast 200 ACK]
    W -. background .-> I[Inbound processor<br/>dedupe wamid · per-user lock]
    I --> E{Engine}
    E -- "bk:confirm / bk:cancel<br/>help · reset" --> B[Booking service<br/>deterministic]
    E -- free text / list taps --> A[Agent loop<br/>LLM + tools]
    A --> T[Tools] --> P[Providers<br/>flights · trains · food]
    T -- draft only --> B
    B -- payment link --> PAY[Checkout page /pay/token]
    PAY -- paid --> B -- receipt + PNR --> M
    A & B & T --> DB[(SQLite/Postgres<br/>contacts · messages · bookings · tool_traces)]
    ADM[/api/* traces · bookings/] --> DB
```

**Agent loop** (`app/agent/agent.py`): user text → the LLM picks tools → results go back to it → repeat (max 8 steps)
→ final reply. Search tools also queue **native WhatsApp list messages**, so the user taps instead of typing.
Tool errors (unknown city, missing address, spend limit) go back to the model as `is_error` results, and it
recovers by asking the user.

### Safety: the LLM can't spend money
1. `prepare_booking` only creates a **draft**. It sends Confirm/Cancel buttons and says so to the model.
2. **Confirm** is a WhatsApp button handled in plain code. The LLM isn't in that path, so a hallucination or a
   prompt injection ("yes, confirm it") can't commit anything.
3. The **user pays on the checkout page themselves**. The payment link is an unguessable token, expires after
   15 minutes, and paying is idempotent.
4. Guardrails: a **per-order spend limit** (`MAX_ORDER_AMOUNT`), **ownership checks** on every booking id that
   comes from a button payload, one open draft at a time (a stale Confirm button is rejected), and waitlisted
   train classes can't be booked.

### Production-grade WhatsApp plumbing
HMAC-SHA256 signature check on webhooks · immediate ACK with background processing · **idempotency** on
`wamid` (Meta retries deliveries) · per-user ordering lock · retries with backoff on 429/5xx · respects the
24-hour customer-service window · delivery status tracking · **mock mode**, so everything runs without Meta.

### Observability
Every tool call is stored in `tool_traces` (input, output, error, latency):
`GET /api/traces?wa_id=91…` shows exactly what the agent did and why. `GET /api/bookings` shows all orders.

## What's real vs mocked (MVP)

| Part | Status | How to go live |
|---|---|---|
| WhatsApp messaging, lists, buttons, CTA links | **Real** (Meta Cloud API) | Already live with your test number |
| AI agent, tool use, memory | **Real** (Gemini / Groq / OpenRouter) | Switch with `LLM_PROVIDER` |
| Flight inventory | Mock, deterministic | Implement `FlightProvider` with Amadeus / Duffel / TBO |
| Train inventory | Mock | IRCTC has no public API, so this needs an authorised agent (B2B) API |
| Food | Mock | ONDC buyer app or a restaurant partner API |
| Payment | Test checkout page + cash on delivery (food) | Swap for a Razorpay Payment Link and call `mark_paid` from its signed webhook |
| Web search, page reading, weather | **Real, live** | Add `TAVILY_API_KEY` for the best search quality |

Each provider sits behind an interface (`app/providers/base.py`), so going live means writing **one class**.
The agent, tools and checkout flow don't change.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-dev.txt
copy .env.example .env                                # set LLM_PROVIDER + LLM_API_KEY (+ Meta creds for live)
uvicorn app.main:app --reload
```

**Without a phone**, use the terminal simulator. It sends correctly signed webhooks, and you type a number to tap a list or button option:
```bash
python scripts/simulate.py --name "Dhruv"
```

**On real WhatsApp** (Meta test number):
1. `ngrok http 8000`, then set `PUBLIC_BASE_URL=https://<id>.ngrok-free.app` in `.env` and restart.
2. Meta App → WhatsApp → Configuration → Webhook: URL `https://<id>.ngrok-free.app/webhook`, verify token
   = `WHATSAPP_VERIFY_TOKEN`, then subscribe to **messages**.
3. Message the test number from a phone you added as a recipient.

**Tests** (offline, with a scripted LLM, so they're deterministic): `pytest`. They cover the full flight / train / food
flows end to end through signed webhooks, the checkout, the spend limit, ownership, stale drafts, tool-error recovery
and memory trimming.

## Switching LLM provider
All three providers expose an OpenAI-compatible API, so a single adapter (`app/agent/llm.py`) covers them. Only `.env` changes:

| `LLM_PROVIDER` | Get a key | Default `LLM_MODEL` |
|---|---|---|
| `gemini` | aistudio.google.com/apikey (free tier) | `gemini-3-flash-preview` |
| `groq` | console.groq.com/keys (free tier, very fast) | `llama-3.3-70b-versatile` |
| `openrouter` | openrouter.ai/keys (any model) | `google/gemini-2.5-flash` |

Set `LLM_MODEL` to use any other tool-calling model from that provider. Set `LLM_BASE_URL` to point at another
OpenAI-compatible endpoint (for example a local Ollama).

## Why no agent framework (and how this maps to LangGraph)

The agent loop is written by hand (about 150 lines in `app/agent/agent.py` + `app/agent/llm.py`) instead of
using LangChain / LangGraph / CrewAI. That was deliberate:

- **Transparent:** every step (tool selection, tool results, memory trimming, error recovery) is visible
  and debuggable. Supporting Gemini 3's thought signatures took a few lines in one adapter.
- **Provider-agnostic:** switching Anthropic → Gemini / Groq / OpenRouter touched a single file.
- **Safety lives outside the model:** Confirm and payment are plain code paths the LLM can't reach, so no
  framework abstraction is needed (or able) to enforce them.
- **Small footprint:** no framework dependencies to pin or upgrade.

The design already follows LangGraph's model, so migrating is a mechanical change, not a rewrite:

| This project | LangGraph equivalent |
|---|---|
| `run_agent()` loop in `agent.py` | `StateGraph` with an **agent** node and a **tools** node, plus a conditional edge "tool calls? → tools : END" |
| `messages` list + `Contact.history` | Graph **state** (`MessagesState`) persisted with a **checkpointer** keyed by `thread_id = wa_id` |
| `TOOLS` schemas + `REGISTRY` in `tools.py` | `@tool` functions bound with `llm.bind_tools(...)`, executed by `ToolNode` |
| `ToolContext` (db session, contact, outbox) | Runtime config / injected state passed to tools |
| `agent_max_steps = 8` | `recursion_limit` in the run config |
| `_trim()` memory window | `trim_messages` (or a summarisation node) before the agent node |
| `ctx.outbox` (WhatsApp lists / buttons) | A state field written by tools and sent by a final **respond** node |
| `remember()` system notes (button taps, payments) | Updating graph state for the thread from outside the graph (`graph.update_state`) |
| `bk:confirm` / `bk:cod` buttons handled in code | A **human-in-the-loop interrupt** (`interrupt()` before the commit step), resumed by the WhatsApp button webhook |
| `tool_traces` table + `/api/traces` | LangSmith tracing (or keep the table via a callback) |
| `OpenAICompatibleLLM` adapter | `ChatOpenAI(base_url=...)` / `ChatGoogleGenerativeAI` / `ChatGroq` |

```python
# Sketch of the same agent in LangGraph (tools and prompt unchanged)
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

graph = StateGraph(MessagesState)
graph.add_node("agent", lambda s: {"messages": [llm.bind_tools(tools).invoke([system, *s["messages"]])]})
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # tool calls -> "tools", else END
graph.add_edge("tools", "agent")
app = graph.compile(checkpointer=SqliteSaver(conn))     # memory per thread_id = WhatsApp number
```

**When I'd switch:** multi-agent setups (for example a separate travel planner and payments agent), long-running
or resumable workflows that wait hours for a user or a webhook, parallel tool branches, or a team that
standardises on LangSmith for tracing and evals.

## Project layout
```
app/
  agent/    agent.py (loop + memory) · llm.py (Gemini/Groq/OpenRouter adapter) · tools.py (tool schemas + impls) · engine.py (routing)
  providers/ base.py (interfaces) · mock.py (flights, trains, food)
  services/ bookings.py (lifecycle, guardrails) · messaging.py (24h window, persistence) · inbound.py
  api/      webhook.py · pay.py (checkout) · admin.py (traces, bookings)
  whatsapp/ client.py (Cloud API + message builders) · parser.py
scripts/simulate.py · tests/
```

## Next steps
Real providers (above) · voice notes via speech-to-text · price-drop alerts (a background job plus an approved template
message) · round trips and multi-city · Redis lock and a queue for multi-instance scale · Postgres.
