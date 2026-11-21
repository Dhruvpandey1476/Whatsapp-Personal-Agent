"""Processes a verified webhook payload. Runs in the background after we have already ACKed Meta."""

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.agent.engine import handle_message
from app.db import SessionLocal, utcnow
from app.models import Contact, Message
from app.services.messaging import apply_status_update, get_client
from app.whatsapp.parser import InboundMessage, parse_webhook

log = logging.getLogger(__name__)

# One lock per contact so two quick messages from the same user are handled in order and don't race
# on the conversation state. (Single-process; with several workers use a Redis lock / partitioned queue.)
_contact_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def process_webhook(payload: dict[str, Any]) -> None:
    messages, statuses = parse_webhook(payload)
    for msg in messages:
        try:
            async with _contact_locks[msg.from_wa_id]:
                await handle_inbound(msg)
        except Exception:  # never let one bad message kill the batch
            log.exception("failed to process inbound message %s", msg.wa_message_id)

    if statuses:
        async with SessionLocal() as db:
            for st in statuses:
                if not await apply_status_update(db, st.wa_message_id, st.status, st.error):
                    log.debug("status for unknown message %s", st.wa_message_id)


async def upsert_contact(db, wa_id: str, name: str | None) -> Contact:
    contact = await db.scalar(select(Contact).where(Contact.wa_id == wa_id))
    if contact is None:
        contact = Contact(wa_id=wa_id, name=name, profile={}, history=[])
        db.add(contact)
        await db.flush()
    elif name and contact.name != name:
        contact.name = name
    return contact


async def handle_inbound(msg: InboundMessage) -> None:
    async with SessionLocal() as db:
        contact = await upsert_contact(db, msg.from_wa_id, msg.profile_name)
        sent_at = datetime.fromtimestamp(msg.timestamp, UTC).replace(tzinfo=None) if msg.timestamp else utcnow()
        contact.last_inbound_at = max(contact.last_inbound_at or sent_at, sent_at)

        # Idempotency: Meta retries webhooks, so the same wamid can arrive more than once.
        db.add(
            Message(
                contact_id=contact.id,
                wa_message_id=msg.wa_message_id,
                direction="in",
                type=msg.type,
                body=msg.text,
                payload=msg.raw,
                status="received",
            )
        )
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            log.info("duplicate webhook for %s ignored", msg.wa_message_id)
            return

        await get_client().mark_read(msg.wa_message_id)
        await handle_message(db, contact, msg)
