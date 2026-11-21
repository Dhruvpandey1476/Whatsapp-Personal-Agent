"""Internal read API (X-API-Key): conversation history, bookings and agent tool traces."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Booking, Contact, Message, ToolTrace
from app.security import require_api_key
from app.services.bookings import summary

router = APIRouter(prefix="/api", tags=["admin"], dependencies=[Depends(require_api_key)])


async def _contact(db, wa_id: str) -> Contact:
    c = await db.scalar(select(Contact).where(Contact.wa_id == wa_id))
    if c is None:
        raise HTTPException(404, "Unknown contact")
    return c


@router.get("/contacts/{wa_id}")
async def contact(wa_id: str) -> dict:
    async with SessionLocal() as db:
        c = await _contact(db, wa_id)
        return {"wa_id": c.wa_id, "name": c.name, "profile": c.profile, "history_turns": len(c.history or [])}


@router.get("/contacts/{wa_id}/messages")
async def messages(wa_id: str, limit: int = 100) -> list[dict]:
    async with SessionLocal() as db:
        c = await _contact(db, wa_id)
        q = select(Message).where(Message.contact_id == c.id).order_by(Message.id.desc()).limit(limit)
        rows = list(await db.scalars(q))
    return [{"id": m.id, "direction": m.direction, "type": m.type, "body": m.body, "status": m.status,
             "created_at": m.created_at.isoformat()} for m in reversed(rows)]  # fmt: skip


@router.get("/messages/{message_id}")
async def message(message_id: int) -> dict:
    async with SessionLocal() as db:
        m = await db.get(Message, message_id)
    if m is None:
        raise HTTPException(404, "Unknown message")
    return {"id": m.id, "direction": m.direction, "type": m.type, "body": m.body, "payload": m.payload, "status": m.status}


@router.get("/bookings")
async def list_bookings(limit: int = 50) -> list[dict]:
    async with SessionLocal() as db:
        rows = await db.scalars(select(Booking).order_by(Booking.id.desc()).limit(limit))
        return [summary(b) | {"details": b.details} for b in rows]


@router.get("/traces")
async def traces(wa_id: str | None = None, limit: int = 50) -> list[dict]:
    """What the agent did and why: every tool call with its input, output and latency."""
    async with SessionLocal() as db:
        q = select(ToolTrace).order_by(ToolTrace.id.desc()).limit(limit)
        if wa_id:
            q = q.where(ToolTrace.contact_id == (await _contact(db, wa_id)).id)
        rows = list(await db.scalars(q))
    return [{"id": t.id, "tool": t.tool, "input": t.input, "output": t.output, "is_error": t.is_error, "ms": t.ms,
             "at": t.created_at.isoformat()} for t in rows]  # fmt: skip
