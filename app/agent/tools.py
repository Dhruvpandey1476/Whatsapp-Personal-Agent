"""Tools the agent can call. Each tool returns JSON-able data for the LLM and may queue
WhatsApp interactive messages (lists / buttons) in `ctx.outbox`, which the engine sends right after
the agent's text reply."""

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Contact
from app.providers import mock
from app.services import bookings
from app.whatsapp.client import buttons_message, list_message

PROFILE_FIELDS = {"full_name", "age", "gender", "email", "home_city", "delivery_address", "preferences"}


@dataclass
class ToolContext:
    db: AsyncSession
    contact: Contact
    outbox: list[dict] = field(default_factory=list)
    _slots: dict[str, int] = field(default_factory=dict)

    def show(self, slot: str, message: dict) -> None:
        """Queue an interactive message. A later message in the same slot replaces the earlier one, so a
        model that searches several times in one turn still sends the user a single list."""
        if slot in self._slots:
            self.outbox[self._slots[slot]] = message
        else:
            self._slots[slot] = len(self.outbox)
            self.outbox.append(message)


class ToolError(Exception):
    """Expected, user-facing failure. Returned to the model as an error tool_result so it can recover."""


def _fmt_dur(m: int) -> str:
    return f"{m // 60}h{m % 60:02d}m"


TOOLS: list[dict] = [
    {
        "name": "search_flights",
        "description": "Search one-way domestic flights in India. Results are also shown to the user as a "
        "tappable WhatsApp list, so in your reply just summarise the best 2-3 options briefly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "City name or IATA code, e.g. Delhi or DEL"},
                "destination": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "adults": {"type": "integer", "minimum": 1, "maximum": 6, "default": 1},
                "sort": {"type": "string", "enum": ["cheapest", "earliest", "fastest"], "default": "cheapest"},
                "depart_after": {"type": "string", "description": "Optional HH:MM filter, e.g. 17:00 for evening"},
                "depart_before": {"type": "string", "description": "Optional HH:MM filter"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "search_trains",
        "description": "Search Indian Railways trains between two cities, with fares and seat availability "
        "per class (1A, 2A, 3A, CC, SL). Results are shown as a tappable list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "City or station code, e.g. Delhi / NDLS"},
                "destination": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "travel_class": {"type": "string", "enum": ["1A", "2A", "3A", "CC", "SL"]},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "search_restaurants",
        "description": "Find restaurants that deliver to the user, by dish or cuisine. Shown as a tappable list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Dish or cuisine, e.g. 'biryani', 'pizza'"},
                "veg_only": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "get_menu",
        "description": "Get a restaurant's menu with item ids and prices.",
        "input_schema": {
            "type": "object",
            "properties": {"restaurant_id": {"type": "string"}},
            "required": ["restaurant_id"],
        },
    },
    {
        "name": "prepare_booking",
        "description": "Create a DRAFT booking/order and send the user Confirm/Cancel buttons. This does NOT "
        "book or charge anything; the user must tap Confirm and then pay via the link themselves. "
        "Flights/trains need offer_id and every passenger's full name, age and gender. "
        "Food needs restaurant_id and items; delivery_address defaults to the saved profile address.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["flight", "train", "food"]},
                "offer_id": {"type": "string", "description": "From search_flights / search_trains results"},
                "passengers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "gender": {"type": "string", "enum": ["M", "F", "O"]},
                            "berth_preference": {"type": "string"},
                        },
                        "required": ["name", "age", "gender"],
                    },
                },
                "restaurant_id": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"item_id": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
                        "required": ["item_id", "qty"],
                    },
                },
                "delivery_address": {"type": "string"},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "get_my_bookings",
        "description": "List the user's recent bookings and orders with status and PNR / order id.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_booking",
        "description": "Cancel a booking by its ref (e.g. FL-AB12CD). Only call after the user clearly asked "
        "to cancel that specific booking.",
        "input_schema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]},
    },
    {
        "name": "update_profile",
        "description": "Save details about the user for future tasks (long-term memory): full_name, age, "
        "gender, email, home_city, delivery_address, preferences (free text, e.g. 'veg, window seat, "
        "prefers IndiGo'). Call whenever the user shares such info. Pass only the fields that changed.",
        "input_schema": {
            "type": "object",
            "properties": {f: {"type": "integer" if f == "age" else "string"} for f in sorted(PROFILE_FIELDS)},
        },
    },
]


# ---------------------------------------------------------------- implementations


async def search_flights(ctx: ToolContext, origin: str, destination: str, date: str, adults: int = 1,
                         sort: str = "cheapest", depart_after: str | None = None,
                         depart_before: str | None = None) -> dict:  # fmt: skip
    results = mock.flights.search(origin, destination, date, adults)
    if depart_after:
        results = [f for f in results if f["depart"] >= depart_after]
    if depart_before:
        results = [f for f in results if f["depart"] <= depart_before]
    key = {"cheapest": "total", "earliest": "depart", "fastest": "duration_min"}.get(sort, "total")
    results = sorted(results, key=lambda f: f[key])[:5]
    if not results:
        return {"results": [], "note": "No flights match those filters. Suggest relaxing the time window."}
    rows = [
        (f"pick:{f['offer_id']}", f"₹{f['total']:,} {f['airline']}",
         f"{f['depart']}→{f['arrive']} · {_fmt_dur(f['duration_min'])} · {'non-stop' if not f['stops'] else '1 stop'}")
        for f in results
    ]  # fmt: skip
    ctx.show("results", list_message(f"✈️ {results[0]['from']} → {results[0]['to']} on {date}", "View flights", rows))
    return {"results": results}


async def search_trains(ctx: ToolContext, origin: str, destination: str, date: str,
                        travel_class: str | None = None) -> dict:  # fmt: skip
    results = mock.trains.search(origin, destination, date, travel_class)
    rows = []
    for t in results:
        for cls, c in t["classes"].items():
            if c["status"].startswith("AVAILABLE"):
                rows.append((f"pick:{c['offer_id']}", f"₹{c['fare']:,} {t['train_no']} {cls}",
                             f"{t['name']} · {t['depart']}→{t['arrive']} · {c['status']}"))  # fmt: skip
    if rows:
        ctx.show("results", list_message(f"🚆 Trains on {date} (available seats only)", "View trains", rows[:10]))
    return {"results": results, "note": None if rows else "No confirmed seats; only waitlist available."}


async def search_restaurants(ctx: ToolContext, query: str = "", veg_only: bool = False) -> dict:
    results = mock.food.search(query, veg_only)
    rows = [(f"rest:{r['id']}", r["name"], f"{'🟢 ' if r['veg'] else ''}{r['cuisine']} · ⭐{r['rating']} · {r['eta_min']} min")
            for r in results]  # fmt: skip
    ctx.show("results", list_message("🍽️ Restaurants near you", "View restaurants", rows))
    return {"results": results}


async def get_menu(ctx: ToolContext, restaurant_id: str) -> dict:
    menu = mock.food.menu(restaurant_id)
    if not menu:
        raise ToolError(f"Unknown restaurant {restaurant_id}")
    return menu


def _passengers(ctx: ToolContext, passengers: list[dict] | None) -> list[dict]:
    if not passengers:
        p = ctx.contact.profile or {}
        if p.get("full_name") and p.get("age") and p.get("gender"):
            passengers = [{"name": p["full_name"], "age": p["age"], "gender": p["gender"]}]
        else:
            raise ToolError("Passenger details missing. Ask the user for full name, age and gender.")
    for x in passengers:
        if not (x.get("name") and x.get("age") and x.get("gender")):
            raise ToolError("Every passenger needs name, age and gender.")
    return passengers


async def prepare_booking(ctx: ToolContext, kind: str, offer_id: str | None = None,
                          passengers: list[dict] | None = None, restaurant_id: str | None = None,
                          items: list[dict] | None = None, delivery_address: str | None = None) -> dict:  # fmt: skip
    if kind == "flight":
        offer = mock.flights.get_offer(offer_id or "")
        if not offer:
            raise ToolError("Offer not found or expired; search again.")
        pax = _passengers(ctx, passengers)
        amount = offer["fare_per_adult"] * len(pax)
        title = f"{offer['airline']} {offer['flight_no']} {offer['from']}→{offer['to']} {offer['date']} {offer['depart']}"
        details = {"offer": offer, "passengers": pax}
        lines = [f"✈️ *{title}*", f"Arrives {offer['arrive']} · {offer['baggage']}"]
    elif kind == "train":
        offer = mock.trains.get_offer(offer_id or "")
        if not offer:
            raise ToolError("Offer not found; search again.")
        if not offer["status"].startswith("AVAILABLE"):
            raise ToolError(f"Class {offer['class']} is {offer['status']}; pick an available class.")
        pax = _passengers(ctx, passengers)
        amount = offer["fare"] * len(pax)
        title = f"{offer['train_no']} {offer['name']} {offer['from']}→{offer['to']} {offer['date']} {offer['class']}"
        details = {"offer": offer, "passengers": pax}
        lines = [f"🚆 *{title}*", f"Departs {offer['depart']}, arrives {offer['arrive']} · {offer['status']}"]
    elif kind == "food":
        menu = mock.food.menu(restaurant_id or "")
        if not menu:
            raise ToolError("Unknown restaurant.")
        if not items:
            raise ToolError("No items given.")
        by_id = {i["item_id"]: i for i in menu["items"]}
        lines_items, subtotal = [], 0
        for it in items:
            m = by_id.get(it["item_id"])
            if not m:
                raise ToolError(f"Item {it['item_id']} is not on {menu['name']}'s menu.")
            subtotal += m["price"] * it["qty"]
            lines_items.append({"name": m["name"], "qty": it["qty"], "price": m["price"]})
        address = delivery_address or (ctx.contact.profile or {}).get("delivery_address")
        if not address:
            raise ToolError("Delivery address missing. Ask the user for it (and offer to save it).")
        fees = 40 + round(subtotal * 0.05)  # delivery + taxes
        amount = subtotal + fees
        title = f"{menu['name']}: " + ", ".join(f"{x['qty']}× {x['name']}" for x in lines_items)
        details = {"restaurant": menu["name"], "items": lines_items, "subtotal": subtotal, "fees": fees,
                   "address": address, "eta_min": menu["eta_min"]}  # fmt: skip
        pax = []
        lines = [f"🍔 *{menu['name']}*", *(f"{x['qty']}× {x['name']} ₹{x['price'] * x['qty']}" for x in lines_items),
                 f"Delivery + taxes ₹{fees}", f"📍 {address}"]  # fmt: skip
    else:
        raise ToolError(f"Unsupported kind {kind}")

    try:
        b = await bookings.create_pending(ctx.db, ctx.contact, kind, title, details, amount)
    except bookings.BookingError as exc:
        raise ToolError(str(exc)) from exc
    if pax:
        lines.append("👤 " + ", ".join(f"{p['name']} ({p['age']}{p['gender']})" for p in pax))
    lines.append(f"\n*Total: ₹{amount:,}*")
    ctx.show(
        "confirm",
        buttons_message("\n".join(lines), [(f"bk:confirm:{b.id}", "✅ Confirm & pay"), (f"bk:cancel:{b.id}", "❌ Cancel")],
                        footer=f"Ref {b.ref} · nothing is charged until you pay")  # fmt: skip
    )
    return {"ref": b.ref, "amount_inr": amount, "status": "awaiting_user_confirmation",
            "note": "Confirm/Cancel buttons were sent. It is NOT booked yet; tell the user to review and tap Confirm."}  # fmt: skip


async def get_my_bookings(ctx: ToolContext) -> dict:
    return {"bookings": [bookings.summary(b) for b in await bookings.list_for_contact(ctx.db, ctx.contact)]}


async def cancel_booking(ctx: ToolContext, ref: str) -> dict:
    try:
        b = await bookings.cancel(ctx.db, ctx.contact, ref)
    except bookings.BookingError as exc:
        raise ToolError(str(exc)) from exc
    refund = b.amount if b.confirmation_code else 0
    return {"ref": b.ref, "status": "cancelled", "refund_inr": refund}


async def update_profile(ctx: ToolContext, **fields: Any) -> dict:
    clean = {k: v for k, v in fields.items() if k in PROFILE_FIELDS and v not in (None, "")}
    ctx.contact.profile = {**(ctx.contact.profile or {}), **clean}  # reassign so SQLAlchemy sees the change
    await ctx.db.commit()
    return {"saved": sorted(clean), "profile": ctx.contact.profile}


REGISTRY = {
    "search_flights": search_flights,
    "search_trains": search_trains,
    "search_restaurants": search_restaurants,
    "get_menu": get_menu,
    "prepare_booking": prepare_booking,
    "get_my_bookings": get_my_bookings,
    "cancel_booking": cancel_booking,
    "update_profile": update_profile,
}
