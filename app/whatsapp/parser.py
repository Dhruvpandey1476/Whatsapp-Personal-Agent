"""Normalise Meta webhook payloads into simple dataclasses.

Payload shape (abridged):
{"object": "whatsapp_business_account",
 "entry": [{"changes": [{"field": "messages", "value": {
     "metadata": {...}, "contacts": [{"wa_id": "...", "profile": {"name": "..."}}],
     "messages": [{"from": "...", "id": "wamid...", "type": "text", "text": {"body": "hi"}}],
     "statuses": [{"id": "wamid...", "status": "delivered", "recipient_id": "..."}]}}]}]}
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InboundMessage:
    wa_message_id: str
    from_wa_id: str
    type: str
    profile_name: str | None = None
    text: str | None = None
    reply_id: str | None = None  # id of the tapped button / list row
    timestamp: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatusUpdate:
    wa_message_id: str
    status: str  # sent | delivered | read | failed
    recipient_id: str | None = None
    error: str | None = None


def _parse_message(msg: dict[str, Any], names: dict[str, str]) -> InboundMessage:
    mtype = msg.get("type", "unknown")
    text: str | None = None
    reply_id: str | None = None

    if mtype == "text":
        text = msg.get("text", {}).get("body")
    elif mtype == "interactive":
        inter = msg.get("interactive", {})
        reply = inter.get("button_reply") or inter.get("list_reply") or {}
        reply_id, text = reply.get("id"), reply.get("title")
    elif mtype == "button":  # quick-reply button on a template message
        reply_id = msg.get("button", {}).get("payload")
        text = msg.get("button", {}).get("text")
    elif mtype in {"image", "video", "document"}:
        text = msg.get(mtype, {}).get("caption")

    sender = msg.get("from", "")
    return InboundMessage(
        wa_message_id=msg["id"],
        from_wa_id=sender,
        type=mtype,
        profile_name=names.get(sender),
        text=text,
        reply_id=reply_id,
        timestamp=int(msg["timestamp"]) if msg.get("timestamp") else None,
        raw=msg,
    )


def parse_webhook(payload: dict[str, Any]) -> tuple[list[InboundMessage], list[StatusUpdate]]:
    messages: list[InboundMessage] = []
    statuses: list[StatusUpdate] = []
    if payload.get("object") != "whatsapp_business_account":
        return messages, statuses

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "messages":
                continue
            value = change.get("value", {})
            names = {
                c.get("wa_id", ""): c.get("profile", {}).get("name")
                for c in value.get("contacts", [])
            }
            for msg in value.get("messages", []):
                if "id" in msg:
                    messages.append(_parse_message(msg, names))
            for st in value.get("statuses", []):
                errors = st.get("errors") or []
                statuses.append(
                    StatusUpdate(
                        wa_message_id=st.get("id", ""),
                        status=st.get("status", ""),
                        recipient_id=st.get("recipient_id"),
                        error="; ".join(e.get("title", str(e)) for e in errors) or None,
                    )
                )
    return messages, statuses
