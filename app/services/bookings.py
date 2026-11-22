"""Booking lifecycle. The LLM can only *prepare* a booking. Committing it requires the user to tap a
real WhatsApp button and then pay on the checkout page themselves, so a hallucination or prompt
injection can never spend money on the user's behalf."""

import secrets
import string
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import utcnow
from app.models import Booking, Contact

PREFIX = {"flight": "FL", "train": "TR", "food": "FD"}
ALNUM = string.ascii_uppercase + string.digits


class BookingError(Exception):
    pass


def _code(n: int, alphabet: str = ALNUM) -> str:
    return "".join(secrets.choice(alphabet) for _ in range(n))


def payment_url(b: Booking) -> str:
    return f"{settings.public_base_url.rstrip('/')}/pay/{b.pay_token}"


async def create_pending(db: AsyncSession, contact: Contact, kind: str, title: str, details: dict, amount: int) -> Booking:
    if amount <= 0:
        raise BookingError("Amount must be positive")
    if amount > settings.max_order_amount:
        raise BookingError(f"Amount ₹{amount} exceeds the safety limit of ₹{settings.max_order_amount}")
    # Only one open draft at a time keeps the confirm buttons unambiguous.
    for old in await db.scalars(
        select(Booking).where(Booking.contact_id == contact.id, Booking.status == "pending_confirmation")
    ):
        old.status = "superseded"
    b = Booking(
        ref=f"{PREFIX[kind]}-{_code(6)}",
        pay_token=secrets.token_urlsafe(24),
        contact_id=contact.id,
        kind=kind,
        title=title[:200],
        details=details,
        amount=amount,
    )
    db.add(b)
    await db.commit()
    return b


async def get_for_contact(db: AsyncSession, contact: Contact, booking_id: int) -> Booking:
    b = await db.get(Booking, booking_id)
    if b is None or b.contact_id != contact.id:  # ownership check: ids come from user-controlled payloads
        raise BookingError("Booking not found")
    return b


async def confirm(db: AsyncSession, contact: Contact, booking_id: int) -> Booking:
    b = await get_for_contact(db, contact, booking_id)
    if b.status == "superseded":
        raise BookingError(f"That option ({b.ref}) was replaced by your newer selection")
    if b.status != "pending_confirmation":
        raise BookingError(f"Booking {b.ref} is already {b.status.replace('_', ' ')}")
    b.status = "awaiting_payment"
    b.updated_at = utcnow()
    await db.commit()
    return b


async def place_cod(db: AsyncSession, contact: Contact, booking_id: int) -> Booking:
    """Cash on delivery (food only): the order is placed now and paid to the delivery partner."""
    b = await get_for_contact(db, contact, booking_id)
    if b.kind != "food":
        raise BookingError("Cash on delivery is only available for food orders")
    if b.status == "superseded":
        raise BookingError(f"That option ({b.ref}) was replaced by your newer selection")
    if b.status != "pending_confirmation":
        raise BookingError(f"Order {b.ref} is already {b.status.replace('_', ' ')}")
    b.status = "confirmed"
    b.details = {**b.details, "payment": "cash_on_delivery"}  # reassign so the JSON change is saved
    b.confirmation_code = "ORD" + _code(7, string.digits)
    await db.commit()
    return b


def expired(b: Booking) -> bool:
    return b.status == "awaiting_payment" and utcnow() - b.updated_at > timedelta(minutes=settings.payment_link_minutes)


async def mark_paid(db: AsyncSession, pay_token: str) -> Booking:
    b = await db.scalar(select(Booking).where(Booking.pay_token == pay_token))
    if b is None:
        raise BookingError("Invalid payment link")
    if b.status == "confirmed":
        return b  # idempotent: double-click / page refresh
    if expired(b):
        b.status = "expired"
        await db.commit()
        raise BookingError("This payment link has expired. Ask the agent to book again.")
    if b.status != "awaiting_payment":
        raise BookingError(f"Booking is {b.status.replace('_', ' ')}")
    b.status = "confirmed"
    b.confirmation_code = {
        "flight": _code(6, string.ascii_uppercase),  # airline PNR
        "train": _code(10, string.digits),  # IRCTC-style 10-digit PNR
        "food": "ORD" + _code(7, string.digits),
    }[b.kind]
    await db.commit()
    return b


async def cancel(db: AsyncSession, contact: Contact, ref_or_id: str | int) -> Booking:
    if isinstance(ref_or_id, int):
        b = await get_for_contact(db, contact, ref_or_id)
    else:
        b = await db.scalar(select(Booking).where(Booking.ref == ref_or_id.upper(), Booking.contact_id == contact.id))
        if b is None:
            raise BookingError(f"No booking {ref_or_id}")
    if b.status in ("cancelled", "expired", "superseded"):
        raise BookingError(f"Booking {b.ref} is already {b.status}")
    if b.kind == "food" and b.status == "confirmed":
        raise BookingError("Food orders can't be cancelled once paid (restaurant is already preparing it)")
    b.status = "cancelled"
    await db.commit()
    return b


async def latest_draft(db: AsyncSession, contact: Contact) -> Booking | None:
    return await db.scalar(
        select(Booking).where(Booking.contact_id == contact.id, Booking.status == "pending_confirmation")
        .order_by(Booking.id.desc()).limit(1)
    )


async def list_for_contact(db: AsyncSession, contact: Contact, limit: int = 10) -> list[Booking]:
    q = select(Booking).where(Booking.contact_id == contact.id).order_by(Booking.id.desc()).limit(limit)
    return list(await db.scalars(q))


def summary(b: Booking) -> dict:
    return {
        "ref": b.ref,
        "kind": b.kind,
        "title": b.title,
        "amount_inr": b.amount,
        "status": "expired" if expired(b) else b.status,
        "confirmation_code": b.confirmation_code,
        "payment": (b.details or {}).get("payment", "online"),
        "created_at": b.created_at.isoformat(timespec="minutes"),
    }
