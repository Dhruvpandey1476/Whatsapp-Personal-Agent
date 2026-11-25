"""Entry point for each inbound WhatsApp message.

Deterministic paths (booking confirm/cancel buttons, commands) are handled in code; everything else
goes to the LLM agent. Money-moving steps never depend on the model's judgement.
"""

import logging
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent import remember, run_agent
from app.agent.llm import Unavailable
from app.agent.tools import ToolContext
from app.config import settings
from app.db import SessionLocal
from app.models import Contact
from app.services import bookings, tts, voice
from app.services.messaging import get_client, send_message
from app.whatsapp.client import audio_message, buttons_message, cta_url_message, text_message
from app.whatsapp.parser import InboundMessage

log = logging.getLogger(__name__)

WELCOME = (
    "👋 Hi {name}! I'm your personal assistant on WhatsApp.\n\n"
    "Just tell me what you need, like:\n"
    "✈️ _Cheapest flight Delhi to Goa next Friday evening_\n"
    "🚆 _3A train Mumbai to Pune tomorrow_\n"
    "🍔 _Order a veg biryani_\n\n"
    "🎙️ You can also send a *voice note*, in English or Hindi.\n\n"
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
    # [title](url) -> "title: url"; drop opaque Google News redirect URLs entirely (unreadable on a phone)
    text = re.sub(r"\[([^\]]+)\]\((https?://news\.google\.com/[^)]+)\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1: \2", text)
    text = re.sub(r"\s*\(?https?://news\.google\.com/\S+", "", text)
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
    elif msg.type == "audio":
        text = await _voice_to_text(db, contact, msg)
        if not text:
            return
        if settings.tts_active:
            # The answer will be spoken, so ask for a short, natural, listenable reply.
            text += ("\n\n[voice note: your reply will be read aloud as a voice message. Keep it short and "
                     "conversational (2-4 sentences), no links, lists, emojis or formatting; say prices in words "
                     "like '3,890 rupees'. Tappable lists and buttons are still shown as usual.]")
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

    wa_id = contact.wa_id  # read before any rollback can expire the object
    try:
        reply = await run_agent(ctx, text)
    except Unavailable:
        log.warning("LLM unavailable for %s", wa_id)
        await _apologise(db, wa_id, "⏳ I'm getting a lot of requests right now. Please try again in a minute.")
        return
    except Exception:
        log.exception("agent failed for %s", wa_id)
        await _apologise(db, wa_id, "😕 Something went wrong on my side. Please try again in a moment.")
        return

    if reply:
        if msg.type == "audio" and settings.tts_active and await _say_voice(db, contact, reply):
            pass  # answered with a voice note
        else:
            for part in _chunks(to_whatsapp(reply)):
                await say(db, contact, text_message(part))
    for interactive in ctx.outbox:
        await say(db, contact, interactive)


async def _apologise(db: AsyncSession, wa_id: str, text: str) -> None:
    """Tell the user something failed, even when the session is broken.

    A rollback expires every loaded object; touching one afterwards makes SQLAlchemy lazy-load it synchronously,
    which async drivers (asyncpg on Postgres) reject with MissingGreenlet. So reload explicitly, and if the
    connection itself is gone, use a brand-new session."""
    try:
        await db.rollback()
        contact = await db.scalar(select(Contact).where(Contact.wa_id == wa_id))
        await say(db, contact, text_message(text))
        return
    except Exception:
        log.exception("apology via current session failed for %s, retrying with a fresh session", wa_id)
    try:
        async with SessionLocal() as fresh:
            contact = await fresh.scalar(select(Contact).where(Contact.wa_id == wa_id))
            await say(fresh, contact, text_message(text))
    except Exception:
        log.exception("could not tell %s about the failure", wa_id)


async def _say_voice(db: AsyncSession, contact: Contact, reply: str) -> bool:
    """Speak the reply as a WhatsApp voice note. Returns False on any failure so the caller sends text instead."""
    try:
        ogg = await tts.synthesize(reply)
        media_id = await get_client().upload_media(ogg, "audio/ogg", "reply.ogg")
        await say(db, contact, {**audio_message(media_id), "_transcript": f"[voice] {to_whatsapp(reply)}"})
        return True
    except Exception:
        log.exception("voice reply failed for %s, sending text instead", contact.wa_id)
        return False


async def _voice_to_text(db: AsyncSession, contact: Contact, msg: InboundMessage) -> str | None:
    """Voice note -> transcript, which is then handled exactly like a typed message."""
    media_id = (msg.raw.get("audio") or {}).get("id")
    if not media_id or not settings.stt_key:
        await say(db, contact, text_message("🎙️ Voice notes aren't enabled yet. Please type your request 🙂"))
        return None
    try:
        audio, mime = await get_client().download_media(media_id)
        text = (await voice.transcribe(audio, mime)).strip()
    except Exception:
        log.exception("voice note failed for %s", contact.wa_id)
        text = ""
    if not text:
        await say(db, contact, text_message("😕 I couldn't catch that voice note. Could you try again or type it?"))
        return None
    return text


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
