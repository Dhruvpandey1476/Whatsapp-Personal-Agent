"""Outbound messaging: every message we send goes through here so it is persisted and policy-checked."""

import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import utcnow
from app.models import Contact, Message
from app.whatsapp.client import WhatsAppAPIError, WhatsAppClient, describe

log = logging.getLogger(__name__)

CUSTOMER_SERVICE_WINDOW = timedelta(hours=24)
STATUS_RANK = {"queued": 0, "sent": 1, "delivered": 2, "read": 3, "failed": 4}

_client: WhatsAppClient | None = None


def get_client() -> WhatsAppClient:
    global _client
    if _client is None:
        _client = WhatsAppClient()
    return _client


def set_client(client: WhatsAppClient) -> None:
    global _client
    _client = client


class PolicyError(Exception):
    """Raised when a send would violate WhatsApp messaging policy."""


def window_open(contact: Contact) -> bool:
    """Free-form messages are only allowed within 24h of the customer's last message."""
    return bool(contact.last_inbound_at and utcnow() - contact.last_inbound_at < CUSTOMER_SERVICE_WINDOW)


async def send_message(
    db: AsyncSession,
    contact: Contact,
    message: dict[str, Any],
    *,
    broadcast_id: int | None = None,
    enforce_policy: bool = True,
) -> Message:
    if enforce_policy:
        # Opt-out blocks proactive (template) messages; replying to someone who just wrote to us is fine.
        if contact.opted_out and message["type"] == "template":
            raise PolicyError("Contact has opted out of marketing/proactive messages")
        if message["type"] != "template" and not window_open(contact):
            raise PolicyError("24h customer-service window is closed; use an approved template message")

    record = Message(
        contact_id=contact.id,
        direction="out",
        type=message["type"],
        body=describe(message),
        payload=message,
        status="queued",
        broadcast_id=broadcast_id,
    )
    db.add(record)
    try:
        record.wa_message_id = await get_client().send(contact.wa_id, message)
        record.status = "sent"
    except WhatsAppAPIError as exc:
        log.error("send to %s failed: %s", contact.wa_id, exc)
        record.status, record.error = "failed", str(exc)
    await db.commit()
    return record


async def apply_status_update(db: AsyncSession, wa_message_id: str, status: str, error: str | None) -> bool:
    msg = await db.scalar(select(Message).where(Message.wa_message_id == wa_message_id))
    if msg is None:
        return False
    # Webhooks can arrive out of order; never move a status backwards (e.g. read -> delivered).
    if STATUS_RANK.get(status, -1) > STATUS_RANK.get(msg.status, -1):
        msg.status = status
        if error:
            msg.error = error
        await db.commit()
    return True
