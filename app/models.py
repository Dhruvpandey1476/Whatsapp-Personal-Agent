from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, utcnow


class Contact(Base):
    """The person who owns this agent. `profile` is long-term memory, `history` short-term memory."""

    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    wa_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(128))
    profile: Mapped[dict] = mapped_column(JSON, default=dict)  # traveller details, address, preferences
    history: Mapped[list] = mapped_column(JSON, default=list)  # recent LLM turns (incl. tool calls)
    opted_out: Mapped[bool] = mapped_column(Boolean, default=False)
    last_inbound_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), index=True)
    wa_message_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    direction: Mapped[str] = mapped_column(String(8))  # in | out
    type: Mapped[str] = mapped_column(String(32))
    body: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="received")
    error: Mapped[str | None] = mapped_column(Text)
    broadcast_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Booking(Base):
    """Lifecycle: pending_confirmation -> awaiting_payment -> confirmed (or cancelled / expired)."""

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(primary_key=True)
    ref: Mapped[str] = mapped_column(String(16), unique=True, index=True)  # e.g. FL-7K2Q9A
    pay_token: Mapped[str] = mapped_column(String(48), unique=True, index=True)  # unguessable checkout id
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), index=True)
    kind: Mapped[str] = mapped_column(String(8))  # flight | train | food
    title: Mapped[str] = mapped_column(String(200))
    details: Mapped[dict] = mapped_column(JSON)
    amount: Mapped[int] = mapped_column(Integer)  # INR
    status: Mapped[str] = mapped_column(String(24), default="pending_confirmation")
    confirmation_code: Mapped[str | None] = mapped_column(String(32))  # PNR / order id after payment
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ToolTrace(Base):
    """Every tool call the agent makes (input, output, latency): observability for the agent loop."""

    __tablename__ = "tool_traces"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(ForeignKey("contacts.id"), index=True)
    tool: Mapped[str] = mapped_column(String(48))
    input: Mapped[dict] = mapped_column(JSON)
    output: Mapped[dict | list | None] = mapped_column(JSON)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
