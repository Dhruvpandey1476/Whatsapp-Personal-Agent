"""Entry point for each inbound WhatsApp message.

Deterministic paths (booking confirm/cancel buttons, commands) are handled in code; everything else
goes to the LLM agent. Money-moving steps never depend on the model's judgement.
"""

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent import remember, run_agent
from app.agent.llm import RateLimited
from app.agent.tools import ToolContext
from app.config import settings
from app.models import Contact
from app.services import bookings
from app.services.messaging import send_message
from app.whatsapp.client import buttons_message, cta_url_message, text_message
from app.whatsapp.parser import InboundMessage

log = logging.getLogger(__name__)

WELCOME = (
    "👋 Hi {name}! I'm your personal assistant on WhatsApp.\n\n"
    "Just tell me what you need, like:\n"
    "✈️ _Cheapest flight Delhi to Goa next Friday evening_\n"
    "🚆 _3A train Mumbai to Pune tomorrow_\n"
    "🍔 _Order a veg biryani_\n\n"
    "I'll find options, you tap to choose, confirm and pay. Type *help* anytime, *reset* to clear our chat."
)


async def say(db: AsyncSession, contact: Contact, message: dict) -> None:
    await send_message(db, contact, message)


def to_whatsapp(text: str) -> str:
    """LLMs default to Markdown; convert it to WhatsApp's formatting."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)  # **bold** -> *bold*
    text = re.sub(r"__(.+?)__", r"_\1_", text)
    text = re.sub(r"^#{1,6}\s*(.+)$", r"*\1*", text, flags=re.M)  # headings -> bold line
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.M)  # bullets
    return text.strip()


def _chunks(text: str, size: int = 4000) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


async def handle_message(db: AsyncSession, contact: Contact, msg: InboundMessage) -> None:
    ctx = ToolContext(db=db, contact=contact)
    first_time = not contact.history and not contact.profile

    if msg.reply_id and msg.reply_id.startswith("bk:"):
        await _booking_button(ctx, msg.reply_id)
        return

    if msg.reply_id and msg.reply_id.startswith("pick:"):
        text = f"I choose this option: {msg.text} (offer_id {msg.reply_id[5:]})"
    elif msg.reply_id and msg.reply_id.startswith("rest:"):
        text = f"Show me the menu of {msg.text} (restaurant_id {msg.reply_id[5:]})"
    elif msg.type == "text" and msg.text:
        text = msg.text.strip()
    else:
        await say(db, contact, text_message("I can only read text messages for now. Please type what you need 🙂"))
        return

    cmd = text.lower()
    if cmd in ("help", "menu", "start") or (first_time and cmd in ("hi", "hello", "hey", "hii")):
        await say(db, contact, text_message(WELCOME.format(name=contact.name or "there")))
        return
    if cmd == "reset":
        contact.history = []
        await db.commit()
        await say(db, contact, text_message("🧹 Chat memory cleared. Your saved profile is kept."))
        return

    if not settings.llm_active:
        await say(db, contact, text_message("⚠️ The AI agent isn't configured yet (LLM_API_KEY missing)."))
        return

    try:
        reply = await run_agent(ctx, text)
    except RateLimited:
        log.warning("LLM rate-limited for %s", contact.wa_id)
        await db.rollback()
        await say(db, contact, text_message("⏳ I'm getting a lot of requests right now. Please try again in a minute."))
        return
    except Exception:
        log.exception("agent failed for %s", contact.wa_id)
        await db.rollback()
        await say(db, contact, text_message("😕 Something went wrong on my side. Please try again in a moment."))
        return

    if reply:
        for part in _chunks(to_whatsapp(reply)):
            await say(db, contact, text_message(part))
    for interactive in ctx.outbox:
        await say(db, contact, interactive)


async def _booking_button(ctx: ToolContext, reply_id: str) -> None:
    db, contact = ctx.db, ctx.contact
    try:
        _, action, raw_id = reply_id.split(":", 2)
        booking_id = int(raw_id)
        if action == "confirm":
            b = await bookings.confirm(db, contact, booking_id)
            await say(db, contact, cta_url_message(
                f"🔒 *{b.title}*\nAmount: *₹{b.amount:,}*\n\nTap below to pay securely. "
                f"The link is valid for {settings.payment_link_minutes} minutes.",
                f"Pay ₹{b.amount:,}", bookings.payment_url(b), footer=f"Ref {b.ref}"))  # fmt: skip
            await remember(ctx, f"user confirmed {b.ref}; payment link sent, waiting for the user to pay")
        elif action == "cod":
            b = await bookings.place_cod(db, contact, booking_id)
            await say(db, contact, text_message(
                f"✅ *Order placed: {b.title}*\nOrder ID: *{b.confirmation_code}*\n"
                f"💵 Pay *₹{b.amount:,}* in cash on delivery\n🛵 Arriving in ~{b.details['eta_min']} min"))  # fmt: skip
            await remember(ctx, f"user placed {b.ref} with cash on delivery; order id {b.confirmation_code}")
        elif action == "cancel":
            b = await bookings.cancel(db, contact, booking_id)
            await say(db, contact, text_message(f"❌ Draft {b.ref} discarded. Anything else I can do?"))
            await remember(ctx, f"user discarded draft {b.ref}")
        else:
            raise bookings.BookingError("Unknown action")
    except (bookings.BookingError, ValueError) as exc:
        current = await bookings.latest_draft(db, contact)
        if current and f":{current.id}" not in reply_id:
            # A stale button was tapped: point the user at the draft that is actually open.
            await say(db, contact, buttons_message(
                f"⚠️ {exc}. Your current selection is:\n*{current.title}*\nTotal *₹{current.amount:,}*",
                [(f"bk:confirm:{current.id}", "✅ Confirm & pay"), (f"bk:cancel:{current.id}", "❌ Cancel")],
                footer=f"Ref {current.ref}"))  # fmt: skip
        else:
            await say(db, contact, text_message(f"⚠️ {exc}"))
