"""Hosted checkout page (test-mode gateway). The user opens the link from WhatsApp and pays themselves.

In production this page is replaced by a Razorpay/Stripe payment link, and `mark_paid` is called from
the gateway's signed payment webhook instead of the form POST below. The rest of the flow is the same.
"""

import html
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select

from app.agent.agent import remember
from app.agent.tools import ToolContext
from app.db import SessionLocal
from app.models import Booking, Contact
from app.services import bookings
from app.services.messaging import PolicyError, send_message
from app.whatsapp.client import text_message

log = logging.getLogger(__name__)
router = APIRouter(tags=["payments"])

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Checkout</title><style>
body{{font-family:system-ui,sans-serif;background:#f2f4f3;margin:0;display:flex;justify-content:center;padding:24px 16px}}
.card{{background:#fff;max-width:420px;width:100%;border-radius:16px;padding:24px;box-shadow:0 4px 20px #0001}}
h1{{font-size:20px;margin:0 0 4px}} .muted{{color:#667;font-size:13px}} .amt{{font-size:32px;font-weight:700;margin:16px 0}}
button{{width:100%;padding:14px;border:0;border-radius:10px;background:#128c7e;color:#fff;font-size:16px;font-weight:600;cursor:pointer}}
.badge{{display:inline-block;background:#fff3cd;color:#856404;border-radius:6px;padding:2px 8px;font-size:12px}}
.ok{{color:#128c7e}} .err{{color:#b00020}} ul{{padding-left:18px}}</style></head>
<body><div class="card">{body}</div></body></html>"""


def _page(body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(PAGE.format(body=body), status_code=status)


def _details(b: Booking) -> str:
    d = b.details
    rows = []
    if b.kind == "food":
        rows = [f"{i['qty']}× {html.escape(i['name'])} — ₹{i['price'] * i['qty']}" for i in d["items"]]
        rows.append(f"Delivery + taxes — ₹{d['fees']}")
        rows.append("📍 " + html.escape(d["address"]))
    else:
        rows = [f"👤 {html.escape(p['name'])} ({p['age']}{p['gender']})" for p in d["passengers"]]
    return "<ul>" + "".join(f"<li>{r}</li>" for r in rows) + "</ul>"


@router.get("/pay/{token}", response_class=HTMLResponse, include_in_schema=False)
async def checkout(token: str) -> HTMLResponse:
    async with SessionLocal() as db:
        b = await db.scalar(select(Booking).where(Booking.pay_token == token))
    if b is None:
        return _page("<h1 class='err'>Invalid payment link</h1>", 404)
    if b.status == "confirmed":
        return _page(f"<h1 class='ok'>✅ Already paid</h1><p>{html.escape(b.title)}</p><p>Code <b>{b.confirmation_code}</b></p>")
    if b.status != "awaiting_payment" or bookings.expired(b):
        return _page(f"<h1 class='err'>Link no longer valid</h1><p class='muted'>Booking is {b.status}. Ask the agent again.</p>", 410)
    return _page(
        f"<span class='badge'>TEST MODE · no real money</span><h1>{html.escape(b.title)}</h1>"
        f"<div class='muted'>Ref {b.ref}</div>{_details(b)}<div class='amt'>₹{b.amount:,}</div>"
        f"<form method='post'><button>Pay ₹{b.amount:,}</button></form>"
    )


@router.post("/pay/{token}", response_class=HTMLResponse, include_in_schema=False)
async def pay(token: str, request: Request) -> HTMLResponse:
    async with SessionLocal() as db:
        try:
            b = await bookings.mark_paid(db, token)
        except bookings.BookingError as exc:
            return _page(f"<h1 class='err'>Payment failed</h1><p>{html.escape(str(exc))}</p>", 400)

        contact = await db.get(Contact, b.contact_id)
        label = {"flight": "PNR", "train": "PNR", "food": "Order ID"}[b.kind]
        extra = f"\n🛵 Arriving in ~{b.details['eta_min']} min" if b.kind == "food" else ""
        try:
            await send_message(db, contact, text_message(
                f"🎉 *Payment received: ₹{b.amount:,}*\n{b.title}\n{label}: *{b.confirmation_code}*{extra}\n\n"
                "Anything else I can help with?"))  # fmt: skip
        except PolicyError:
            log.warning("could not notify %s: 24h window closed", contact.wa_id)
        await remember(ToolContext(db=db, contact=contact), f"user paid for {b.ref}; it is confirmed, {label} {b.confirmation_code}")

    return _page(f"<h1 class='ok'>✅ Payment successful</h1><p>{html.escape(b.title)}</p>"
                 f"<p>{label}: <b>{b.confirmation_code}</b></p><p class='muted'>You can go back to WhatsApp.</p>")  # fmt: skip
