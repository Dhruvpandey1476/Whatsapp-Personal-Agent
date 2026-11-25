"""End-to-end tests: signed webhook -> agent loop (scripted LLM) -> tools -> WhatsApp messages -> checkout."""

import json
from datetime import date, timedelta

import pytest
from sqlalchemy import select

from app.agent.agent import set_llm
from app.db import SessionLocal
from app.models import Booking, Contact, ToolTrace
from app.providers import mock
from tests.conftest import last_out, outbound, post_webhook, reply_payload, row_ids, text_payload

D = (date.today() + timedelta(days=7)).isoformat()


class ScriptedLLM:
    """Plays a fixed list of assistant turns and records what it was sent."""

    def __init__(self, *turns):
        self.turns, self.calls = list(turns), []

    async def create(self, *, system, tools, messages):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        return self.turns.pop(0)


def tool(name, **inp):
    return {"content": [{"type": "tool_use", "id": f"tu_{name}_{len(inp)}", "name": name, "input": inp}],
            "stop_reason": "tool_use"}  # fmt: skip


def say(text):
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


async def bookings():
    async with SessionLocal() as db:
        return list(await db.scalars(select(Booking)))


def body(m):
    return m["text"]["body"] if m["type"] == "text" else m["interactive"]["body"]["text"]


# ---------------------------------------------------------------- providers


def test_city_resolution_and_determinism():
    assert mock.resolve_city("new delhi") == "DEL" and mock.resolve_city("NDLS") == "DEL"
    assert mock.resolve_city("Bangalore") == "BLR" and mock.resolve_city("Atlantis") is None
    a, b = mock.flights.search("Delhi", "Goa", D), mock.flights.search("DEL", "GOI", D)
    assert a == b and a[0]["total"] <= a[-1]["total"]
    assert mock.flights.get_offer(a[0]["offer_id"]) == a[0]
    t = mock.trains.search("Mumbai", "Pune", D)[0]
    oid = next(iter(t["classes"].values()))["offer_id"]
    assert mock.trains.get_offer(oid)["offer_id"] == oid


# ---------------------------------------------------------------- full flows


async def test_flight_search_confirm_and_user_pays(client):
    offer = mock.flights.search("DEL", "GOI", D)[0]
    llm = ScriptedLLM(
        tool("search_flights", origin="Delhi", destination="Goa", date=D),
        say("Cheapest is IndiGo, tap to pick 👇"),
        tool("prepare_booking", kind="flight", offer_id=offer["offer_id"],
             passengers=[{"name": "Dhruv Pandey", "age": 24, "gender": "M"}]),
        say("Review and tap Confirm."),
    )  # fmt: skip
    set_llm(llm)

    await post_webhook(client, text_payload("cheapest flight delhi to goa next week"))
    msgs = outbound()
    assert body(msgs[0]).startswith("Cheapest")
    assert f"pick:{offer['offer_id']}" in row_ids(msgs[1])  # results shown as a tappable list

    await post_webhook(client, reply_payload(f"pick:{offer['offer_id']}", "₹4,280 IndiGo"))
    assert offer["offer_id"] in llm.calls[-1]["messages"][-3]["content"]  # the tap was passed to the agent
    confirm = last_out()
    [draft] = await bookings()
    assert draft.status == "pending_confirmation" and draft.amount == offer["fare_per_adult"]
    assert row_ids(confirm) == [f"bk:confirm:{draft.id}", f"bk:cancel:{draft.id}"]

    # Confirm button is handled in code (no LLM call) and returns a payment link, nothing charged yet
    n_llm = len(llm.calls)
    await post_webhook(client, reply_payload(f"bk:confirm:{draft.id}", kind="button_reply"))
    assert len(llm.calls) == n_llm
    link = last_out()["interactive"]
    assert link["type"] == "cta_url"
    url = link["action"]["parameters"]["url"]
    assert url == f"http://test/pay/{draft.pay_token}"
    assert (await bookings())[0].status == "awaiting_payment"

    # The user pays on the checkout page themselves
    page = await client.get(f"/pay/{draft.pay_token}")
    assert page.status_code == 200 and "TEST MODE" in page.text
    paid = await client.post(f"/pay/{draft.pay_token}")
    assert paid.status_code == 200 and "Payment successful" in paid.text
    [b] = await bookings()
    assert b.status == "confirmed" and len(b.confirmation_code) == 6
    assert b.confirmation_code in body(last_out())  # WhatsApp receipt with PNR

    # paying twice is idempotent
    await client.post(f"/pay/{draft.pay_token}")
    assert (await bookings())[0].confirmation_code == b.confirmation_code

    # the agent's memory knows about the payment for the next turn
    async with SessionLocal() as db:
        c = await db.scalar(select(Contact))
        assert "paid" in c.history[-1]["content"]


async def test_train_uses_saved_profile_for_passenger(client):
    t = next(t for t in mock.trains.search("BOM", "PNQ", D) if any(c["status"].startswith("AVAIL") for c in t["classes"].values()))
    oid = next(c["offer_id"] for c in t["classes"].values() if c["status"].startswith("AVAIL"))
    set_llm(ScriptedLLM(
        tool("update_profile", full_name="Dhruv Pandey", age=24, gender="M"),
        tool("prepare_booking", kind="train", offer_id=oid),  # no passengers: taken from profile
        say("Done, please confirm."),
    ))  # fmt: skip
    await post_webhook(client, text_payload("I'm Dhruv Pandey, 24, male. book that train"))
    [b] = await bookings()
    assert b.details["passengers"] == [{"name": "Dhruv Pandey", "age": 24, "gender": "M"}]
    assert "Dhruv Pandey (24M)" in body(last_out())


async def test_food_order_needs_address_then_succeeds(client):
    llm = ScriptedLLM(
        tool("prepare_booking", kind="food", restaurant_id="R1", items=[{"item_id": "R1-2", "qty": 2}]),
        say("What's your delivery address?"),
        tool("update_profile", delivery_address="12 MG Road, Bengaluru"),
        tool("prepare_booking", kind="food", restaurant_id="R1", items=[{"item_id": "R1-2", "qty": 2}]),
        say("Order ready to confirm."),
    )
    set_llm(llm)
    await post_webhook(client, text_payload("2 veg biryani from biryani blues"))
    # the tool error went back to the model as an is_error tool_result, so it could ask the user
    tr = llm.calls[1]["messages"][-1]["content"][0]
    assert tr["is_error"] and "address" in tr["content"]
    assert await bookings() == []

    await post_webhook(client, text_payload("12 MG Road, Bengaluru"))
    [b] = await bookings()
    assert b.amount == 2 * 249 + 40 + round(498 * 0.05)
    await post_webhook(client, reply_payload(f"bk:confirm:{b.id}", kind="button_reply"))
    await client.post(f"/pay/{b.pay_token}")
    assert (await bookings())[0].confirmation_code.startswith("ORD")


# ---------------------------------------------------------------- guardrails


async def test_spend_limit_blocks_expensive_drafts(client):
    offer = mock.flights.search("DEL", "COK", D, adults=6)[-1]
    set_llm(ScriptedLLM(
        tool("prepare_booking", kind="flight", offer_id=offer["offer_id"],
             passengers=[{"name": f"P{i}", "age": 30, "gender": "M"} for i in range(6)]),
        say("That's above your limit."),
    ))  # fmt: skip
    await post_webhook(client, text_payload("book for 6"))
    if offer["fare_per_adult"] * 6 > 25000:
        assert await bookings() == []
        async with SessionLocal() as db:
            [t] = list(await db.scalars(select(ToolTrace)))
        assert t.is_error and "limit" in t.output["error"]


async def test_cannot_confirm_someone_elses_booking(client):
    set_llm(ScriptedLLM(tool("prepare_booking", kind="food", restaurant_id="R3", items=[{"item_id": "R3-1", "qty": 1}],
                             delivery_address="x"), say("ok")))  # fmt: skip
    await post_webhook(client, text_payload("dosa"))
    [b] = await bookings()
    await post_webhook(client, reply_payload(f"bk:confirm:{b.id}", kind="button_reply", wa_id="910000000000"))
    assert (await bookings())[0].status == "pending_confirmation"
    assert "not found" in body(last_out())


async def test_new_draft_supersedes_old_and_stale_confirm_rejected(client):
    p = dict(kind="food", restaurant_id="R3", items=[{"item_id": "R3-1", "qty": 1}], delivery_address="x")
    set_llm(ScriptedLLM(tool("prepare_booking", **p), say("ok"), tool("prepare_booking", **p), say("ok")))
    await post_webhook(client, text_payload("dosa"))
    await post_webhook(client, text_payload("actually make it again"))
    old, new = sorted(await bookings(), key=lambda b: b.id)
    assert old.status == "superseded" and new.status == "pending_confirmation"
    await post_webhook(client, reply_payload(f"bk:confirm:{old.id}", kind="button_reply"))
    # stale tap: explains and re-offers the buttons for the draft that is actually open
    assert "replaced by your newer selection" in body(last_out())
    assert row_ids(last_out()) == [f"bk:confirm:{new.id}", f"bk:cancel:{new.id}"]


async def test_invalid_and_unpaid_links(client):
    assert (await client.get("/pay/nope")).status_code == 404
    set_llm(ScriptedLLM(tool("prepare_booking", kind="food", restaurant_id="R3", items=[{"item_id": "R3-1", "qty": 1}],
                             delivery_address="x"), say("ok")))  # fmt: skip
    await post_webhook(client, text_payload("dosa"))
    [b] = await bookings()
    # not confirmed on WhatsApp yet, so the link can't be paid
    assert (await client.post(f"/pay/{b.pay_token}")).status_code == 400


async def test_bad_tool_args_are_recoverable(client):
    llm = ScriptedLLM(tool("search_flights", origin="Atlantis", destination="Goa", date=D), say("Which city?"))
    set_llm(llm)
    await post_webhook(client, text_payload("flight from atlantis"))
    assert llm.calls[1]["messages"][-1]["content"][0]["is_error"]
    assert body(last_out()) == "Which city?"


async def test_commands_skip_llm(client):
    set_llm(ScriptedLLM())  # would raise if called
    await post_webhook(client, text_payload("hi"))
    assert "personal assistant" in body(last_out())
    await post_webhook(client, text_payload("reset"))
    assert "cleared" in body(last_out())


async def test_history_is_trimmed_without_orphan_tool_results():
    from app.agent.agent import _trim

    h = [{"role": "user", "content": "a"}, {"role": "assistant", "content": [{"type": "tool_use"}]},
         {"role": "user", "content": [{"type": "tool_result"}]}, {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}]  # fmt: skip
    assert _trim(h, 4)[0] == {"role": "user", "content": "c"}
    assert _trim(h, 5) == h


def test_markdown_is_converted_to_whatsapp_format():
    from app.agent.engine import to_whatsapp

    assert to_whatsapp("## Options\n- **IndiGo** ₹4,280") == "*Options*\n• *IndiGo* ₹4,280"


async def test_repeated_searches_in_one_turn_send_one_list(client):
    set_llm(ScriptedLLM(tool("search_restaurants", query="pizza"), tool("search_restaurants", query="capsicum"),
                        tool("search_restaurants", query="italian"), say("Here you go")))  # fmt: skip
    await post_webhook(client, text_payload("capsicum pizza"))
    lists = [m for m in outbound() if m["type"] == "interactive"]
    assert len(lists) == 1


def test_overnight_arrival_is_marked():
    assert mock._hhmm(3 * 60 + 40 + 25 * 60) == "04:40 (+1)"
    assert mock._hhmm(9 * 60) == "09:00"


async def test_food_cash_on_delivery(client):
    set_llm(ScriptedLLM(tool("prepare_booking", kind="food", restaurant_id="R3", items=[{"item_id": "R3-1", "qty": 2}],
                             delivery_address="G-257 Akash Nagar"), say("Choose how to pay")))  # fmt: skip
    await post_webhook(client, text_payload("2 masala dosa"))
    [b] = await bookings()
    assert row_ids(last_out()) == [f"bk:confirm:{b.id}", f"bk:cod:{b.id}", f"bk:cancel:{b.id}"]
    await post_webhook(client, reply_payload(f"bk:cod:{b.id}", kind="button_reply"))
    [b] = await bookings()
    assert b.status == "confirmed" and b.details["payment"] == "cash_on_delivery"
    assert "cash on delivery" in body(last_out()) and b.confirmation_code in body(last_out())
    await post_webhook(client, reply_payload(f"bk:cod:{b.id}", kind="button_reply"))  # double tap is rejected
    assert "already confirmed" in body(last_out())


async def test_cod_not_allowed_for_flights(client):
    offer = mock.flights.search("DEL", "BOM", D)[0]
    set_llm(ScriptedLLM(tool("prepare_booking", kind="flight", offer_id=offer["offer_id"],
                             passengers=[{"name": "A", "age": 30, "gender": "M"}]), say("ok")))  # fmt: skip
    await post_webhook(client, text_payload("book"))
    [b] = await bookings()
    assert len(row_ids(last_out())) == 2  # no COD button
    await post_webhook(client, reply_payload(f"bk:cod:{b.id}", kind="button_reply"))
    assert (await bookings())[0].status == "pending_confirmation"


def test_international_flights():
    f = mock.flights.search("Delhi", "Chicago", D)
    assert f and f[0]["to"] == "ORD" and f[0]["international"] and f[0]["total"] > 30000
    assert mock.resolve_airport("maldives") == "MLE"
    with pytest.raises(ValueError, match="within India"):
        mock.trains.search("Delhi", "Dubai", D)


def test_weather_place_selection():
    from app.services.web import REGIONS, pick_place

    hyd = [{"name": "Hyderabad", "country": "Pakistan", "country_code": "PK", "population": 1386330},
           {"name": "Hyderabad", "country": "India", "country_code": "IN", "population": 3597816}]  # fmt: skip
    dubai = [{"name": "Dubai", "country": "United Arab Emirates", "country_code": "AE", "population": 3790000},
             {"name": "Dubai", "country": "India", "country_code": "IN", "population": 2139}]  # fmt: skip
    assert pick_place(hyd)["country_code"] == "IN"
    assert pick_place(dubai)["country_code"] == "AE"  # a village in UP must not win
    assert pick_place(hyd, "Pakistan")["country_code"] == "PK"
    assert REGIONS["goa"]["country"] == "India"


def test_markdown_links_are_made_whatsapp_friendly():
    from app.agent.engine import to_whatsapp

    out = to_whatsapp("See [29 best thrillers](https://news.google.com/rss/articles/CBMi123?oc=5) and "
                      "[IMDb list](https://www.imdb.com/list/ls1)")
    assert out == "See 29 best thrillers and IMDb list: https://www.imdb.com/list/ls1"
    assert "news.google.com" not in to_whatsapp("More: https://news.google.com/rss/articles/abc?oc=5")


async def test_voice_note_is_transcribed_echoed_and_sent_to_agent(client, monkeypatch):
    from app.agent import engine
    from app.config import settings
    from app.services.messaging import get_client

    async def fake_download(media_id):
        assert media_id == "MEDIA123"
        return b"OggS-fake", "audio/ogg; codecs=opus"

    async def fake_transcribe(audio, mime):
        return "kal Delhi se Goa ki sasti flight"

    monkeypatch.setattr(settings, "stt_api_key", "test")
    monkeypatch.setattr(get_client(), "download_media", fake_download)
    monkeypatch.setattr(engine.voice, "transcribe", fake_transcribe)
    llm = ScriptedLLM(say("Searching flights for you"))
    set_llm(llm)

    payload = text_payload("x")
    m = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    m.pop("text"), m.update(type="audio", audio={"id": "MEDIA123", "mime_type": "audio/ogg; codecs=opus", "voice": True})
    await post_webhook(client, payload)

    out = [body(x) for x in outbound()]
    assert llm.calls[0]["messages"][-1]["content"].startswith("kal Delhi se Goa ki sasti flight")
    assert out == ["Searching flights for you"]  # only the answer; the transcript is not echoed back


async def test_voice_note_without_stt_key_asks_to_type(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "stt_api_key", "")
    set_llm(ScriptedLLM())  # must not be called
    payload = text_payload("x")
    m = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    m.pop("text"), m.update(type="audio", audio={"id": "M1"})
    await post_webhook(client, payload)
    assert "type your request" in body(last_out())


def test_postgres_urls_are_made_async():
    from app.db import async_url

    assert async_url("postgres://u:p@h:5432/db") == "postgresql+asyncpg://u:p@h:5432/db"
    assert async_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert async_url("sqlite+aiosqlite:///./agent.db") == "sqlite+aiosqlite:///./agent.db"


def _voice_payload(media_id="MEDIA1"):
    payload = text_payload("x")
    m = payload["entry"][0]["changes"][0]["value"]["messages"][0]
    m.pop("text"), m.update(type="audio", audio={"id": media_id, "mime_type": "audio/ogg; codecs=opus", "voice": True})
    return payload


def _voice_setup(monkeypatch, transcript="kal Goa ki flight"):
    from app.agent import engine
    from app.config import settings
    from app.services.messaging import get_client

    async def fake_download(media_id):
        return b"OggS-in", "audio/ogg"

    async def fake_transcribe(audio, mime):
        return transcript

    monkeypatch.setattr(settings, "stt_api_key", "test")
    monkeypatch.setattr(get_client(), "download_media", fake_download)
    monkeypatch.setattr(engine.voice, "transcribe", fake_transcribe)
    return engine, settings


async def test_voice_in_voice_out(client, monkeypatch):
    engine, settings = _voice_setup(monkeypatch)
    monkeypatch.setattr(settings, "tts_api_key", "test")
    monkeypatch.setattr(settings, "tts_enabled", True)
    spoken = []

    async def fake_synth(text):
        spoken.append(text)
        return b"OggS-reply"

    monkeypatch.setattr(engine.tts, "synthesize", fake_synth)
    llm = ScriptedLLM(say("IndiGo at 6:10 for 3,890 rupees. Shall I book it?"))
    set_llm(llm)
    await post_webhook(client, _voice_payload())

    sent = outbound()
    assert [m["type"] for m in sent] == ["audio"]  # voice only, no duplicate text
    assert sent[0]["audio"]["id"].startswith("media.mock.") and "_transcript" not in sent[0]
    assert spoken == ["IndiGo at 6:10 for 3,890 rupees. Shall I book it?"]
    assert "read aloud" in llm.calls[0]["messages"][-1]["content"]  # agent told to keep it listenable


async def test_voice_reply_falls_back_to_text(client, monkeypatch):
    engine, settings = _voice_setup(monkeypatch)
    monkeypatch.setattr(settings, "tts_api_key", "test")
    monkeypatch.setattr(settings, "tts_enabled", True)

    async def broken_synth(text):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(engine.tts, "synthesize", broken_synth)
    set_llm(ScriptedLLM(say("Here are your flights")))
    await post_webhook(client, _voice_payload())
    assert [m["type"] for m in outbound()] == ["text"] and body(last_out()) == "Here are your flights"


async def test_typed_message_never_gets_voice(client, monkeypatch):
    from app.agent import engine
    from app.config import settings

    monkeypatch.setattr(settings, "tts_api_key", "test")
    monkeypatch.setattr(settings, "tts_enabled", True)

    async def must_not_run(text):
        raise AssertionError("TTS used for a typed message")

    monkeypatch.setattr(engine.tts, "synthesize", must_not_run)
    set_llm(ScriptedLLM(say("Hello!")))
    await post_webhook(client, text_payload("hello there"))
    assert [m["type"] for m in outbound()] == ["text"]


def test_speakable_text():
    from app.services.tts import speakable

    out = speakable("*IndiGo* ✈️ ₹3,890\n• Book: https://x.com/pay/abc")
    assert out == "IndiGo rupees 3,890\nBook:"


async def test_user_still_gets_apology_when_agent_fails_and_session_is_expired(client, monkeypatch):
    """Regression (seen on Render/Postgres): after a failure the session is rolled back, which expires the
    contact; touching it again lazy-loads synchronously -> MissingGreenlet, and the user got no reply at all."""
    from app.agent import engine

    async def failing_agent(ctx, text):
        ctx.db.expire(ctx.contact)  # what a rollback does to every loaded object
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "run_agent", failing_agent)
    await post_webhook(client, text_payload("what's the weather?"))
    assert "Something went wrong" in body(last_out())


def test_reply_language_follows_latest_message():
    from app.agent.agent import reply_language

    assert reply_language("give me the cheapest flight available from delhi to mumbai for today") == "English"
    assert reply_language("show my bookings") == "English"
    assert reply_language("what is the weather in goa") == "English"
    assert reply_language("kal Delhi se Goa ki sasti flight dikhao").startswith("Hinglish")
    assert reply_language("mujhe biryani chahiye").startswith("Hinglish")
    assert reply_language("कल दिल्ली से गोवा की फ्लाइट").startswith("Hindi")


async def test_english_message_tells_model_to_answer_in_english(client):
    llm = ScriptedLLM(say("ok"))
    set_llm(llm)
    await post_webhook(client, text_payload("kal Delhi se Goa ki sasti flight dikhao"))
    assert "latest message is in Hinglish" in llm.calls[-1]["system"]
    llm.turns.append(say("ok"))
    await post_webhook(client, text_payload("give me the cheapest flight from delhi to mumbai for today"))
    assert "latest message is in English" in llm.calls[-1]["system"]


async def test_language_hint_is_next_to_newest_content_and_not_stored(client):
    llm = ScriptedLLM(tool("search_restaurants", query="pizza"), say("Here are pizza places"))
    set_llm(llm)
    await post_webhook(client, text_payload("show me pizza places nearby"))
    first, second = llm.calls[0]["messages"][-1], llm.calls[1]["messages"][-1]
    assert first["content"].endswith("[Reply language: English. Ignore the language of earlier replies.]")
    assert second["content"][-1]["text"].startswith("[Reply language: English")  # after the tool result too
    async with SessionLocal() as db:
        c = await db.scalar(select(Contact))
    assert "Reply language" not in json.dumps(c.history)  # hint never saved


async def test_old_tool_results_are_shortened_in_history(client):
    set_llm(ScriptedLLM(tool("search_restaurants", query=""), say("Here you go")))
    await post_webhook(client, text_payload("restaurants"))
    async with SessionLocal() as db:
        c = await db.scalar(select(Contact))
    [result] = [b for m in c.history if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]
    assert len(result["content"]) < 400 and result["content"].endswith("(older result, shortened)")
