"""End-to-end tests: signed webhook -> agent loop (scripted LLM) -> tools -> WhatsApp messages -> checkout."""

from datetime import date, timedelta

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
    assert old.status == "cancelled" and new.status == "pending_confirmation"
    await post_webhook(client, reply_payload(f"bk:confirm:{old.id}", kind="button_reply"))
    assert "already cancelled" in body(last_out())


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
