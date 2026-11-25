"""SkyTrip: a small demo airline-booking website the browser agent operates on.

It behaves like a real site (OTP login, session cookie, search, passenger form, review, "Book now"), so the
browser agent can be demonstrated end to end without scraping a real airline, which would break their terms
and hit CAPTCHAs. Real sites don't send OTPs over WhatsApp; here the code is delivered on WhatsApp as a stand-in
for SMS. State is in memory: it's a demo.
"""

import html
import secrets
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Contact
from app.providers import mock
from app.services.messaging import PolicyError, send_message
from app.whatsapp.client import text_message

router = APIRouter(prefix="/skytrip", include_in_schema=False)
COOKIE = "skytrip_session"
OTP_TTL = 300
MAX_OTP_TRIES = 5

_otps: dict[str, dict] = {}  # phone -> {"code", "expires", "tries"}
_sessions: dict[str, dict] = {}  # session id -> {"phone", "draft"}
_trips: dict[str, list[dict]] = {}  # phone -> bookings

CSS = """
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}
header{background:#0c4a6e;color:#fff;padding:14px 20px;display:flex;justify-content:space-between;align-items:center}
header a{color:#bae6fd;margin-left:14px}
main{max-width:640px;margin:24px auto;background:#fff;padding:24px;border-radius:14px;box-shadow:0 2px 12px #0001}
label{display:block;margin-top:12px;font-weight:600} input,select{width:100%;padding:10px;margin-top:4px;
border:1px solid #cbd5e1;border-radius:8px;font-size:15px;box-sizing:border-box}
button{margin-top:16px;padding:11px 18px;border:0;border-radius:8px;background:#0284c7;color:#fff;font-size:15px;
cursor:pointer} .danger{background:#c2410c} .muted{color:#64748b;font-size:13px}
.card{border:1px solid #e2e8f0;border-radius:10px;padding:12px;margin:10px 0}
.badge{background:#fef3c7;color:#92400e;border-radius:6px;padding:2px 8px;font-size:12px}
.err{color:#b91c1c} .ok{color:#15803d}
"""


def _page(title: str, body: str, phone: str | None = None, status: int = 200) -> HTMLResponse:
    nav = (f'<span>Signed in as +{html.escape(phone)}<a href="/skytrip/search">Search</a>'
           f'<a href="/skytrip/trips">My trips</a><a href="/skytrip/logout">Log out</a></span>') if phone else ""  # fmt: skip
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>{html.escape(title)} · SkyTrip</title><style>{CSS}</style></head><body>"
        f"<header><b>✈ SkyTrip</b> <span class='badge'>demo site</span>{nav}</header>"
        f"<main><h2>{html.escape(title)}</h2>{body}</main></body></html>",
        status_code=status,
    )


def _phone(raw: str) -> str:
    digits = "".join(ch for ch in raw if ch.isdigit())
    return "91" + digits if len(digits) == 10 else digits


def _user(request: Request) -> tuple[str | None, dict | None]:
    sess = _sessions.get(request.cookies.get(COOKIE, ""))
    return (sess["phone"], sess) if sess else (None, None)


def _login_redirect() -> RedirectResponse:
    return RedirectResponse("/skytrip/login", status_code=303)


async def _deliver_otp(phone: str, code: str) -> bool:
    """Send the code on WhatsApp (stand-in for SMS). Only to numbers that have chatted with the assistant."""
    async with SessionLocal() as db:
        contact = await db.scalar(select(Contact).where(Contact.wa_id == phone))
        if contact is None:
            return False
        try:
            await send_message(db, contact, text_message(
                f"🔐 *SkyTrip login code: {code}*\nValid for 5 minutes.\n\n"
                "_Demo: a real site would send this by SMS. Only share it with your assistant if you want it to "
                "log in for you._"))  # fmt: skip
        except PolicyError:
            return False
    return True


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    phone, _ = _user(request)
    return RedirectResponse("/skytrip/search" if phone else "/skytrip/login", status_code=303)


@router.get("/login", response_class=HTMLResponse)
async def login_page(error: str = ""):
    err = f"<p class='err'>{html.escape(error)}</p>" if error else ""
    return _page("Log in", f"""{err}<p>Log in with your mobile number. We'll send you a one-time code.</p>
<form method="post" action="/skytrip/login/send">
<label for="phone">Mobile number</label><input id="phone" name="phone" placeholder="e.g. 9810012345" required>
<button type="submit">Send code</button></form>""")


@router.post("/login/send", response_class=HTMLResponse)
async def send_code(phone: str = Form(...)):
    p = _phone(phone)
    if len(p) < 10:
        return await login_page("Enter a valid mobile number.")
    code = f"{secrets.randbelow(1_000_000):06d}"
    _otps[p] = {"code": code, "expires": time.time() + OTP_TTL, "tries": 0}
    if not await _deliver_otp(p, code):
        return await login_page("We couldn't send a code to that number. (Demo: it must have chatted with the assistant.)")
    return _page("Enter code", f"""<p class='ok'>We sent a 6-digit code to +{html.escape(p)}.</p>
<form method="post" action="/skytrip/login/verify"><input type="hidden" name="phone" value="{html.escape(p)}">
<label for="otp">One-time code</label><input id="otp" name="otp" inputmode="numeric" placeholder="6-digit code" required>
<button type="submit">Verify and log in</button></form>""")


@router.post("/login/verify", response_class=HTMLResponse)
async def verify(phone: str = Form(...), otp: str = Form(...)):
    p, entry = _phone(phone), _otps.get(_phone(phone))
    if not entry or time.time() > entry["expires"]:
        return await login_page("Code expired. Please request a new one.")
    entry["tries"] += 1
    if entry["tries"] > MAX_OTP_TRIES:
        _otps.pop(p, None)
        return await login_page("Too many wrong attempts. Request a new code.")
    if not secrets.compare_digest(otp.strip(), entry["code"]):
        return _page("Enter code", f"""<p class='err'>Wrong code, please try again.</p>
<form method="post" action="/skytrip/login/verify"><input type="hidden" name="phone" value="{html.escape(p)}">
<label for="otp">One-time code</label><input id="otp" name="otp" inputmode="numeric" required>
<button type="submit">Verify and log in</button></form>""", status=400)
    _otps.pop(p, None)
    sid = secrets.token_urlsafe(24)
    _sessions[sid] = {"phone": p, "draft": None}
    resp = RedirectResponse("/skytrip/search", status_code=303)
    resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", max_age=7 * 24 * 3600)
    return resp


@router.get("/logout")
async def logout(request: Request):
    _sessions.pop(request.cookies.get(COOKIE, ""), None)
    resp = _login_redirect()
    resp.delete_cookie(COOKIE)
    return resp


@router.get("/search", response_class=HTMLResponse)
async def search(request: Request):
    phone, _ = _user(request)
    if not phone:
        return _login_redirect()
    return _page("Search flights", """<form method="get" action="/skytrip/results">
<label for="origin">From</label><input id="origin" name="origin" placeholder="City, e.g. Delhi" required>
<label for="destination">To</label><input id="destination" name="destination" placeholder="City, e.g. Mumbai" required>
<label for="date">Date</label><input id="date" name="date" placeholder="YYYY-MM-DD" required>
<button type="submit">Search flights</button></form>""", phone)


@router.get("/results", response_class=HTMLResponse)
async def results(request: Request, origin: str, destination: str, date: str):
    phone, _ = _user(request)
    if not phone:
        return _login_redirect()
    try:
        flights = mock.flights.search(origin, destination, date)
    except ValueError as exc:
        return _page("Search flights", f"<p class='err'>{html.escape(str(exc))}</p><a href='/skytrip/search'>Back</a>", phone)
    rows = "".join(
        f"""<div class="card"><b>{html.escape(f['airline'])} {f['flight_no']}</b> · {f['depart']} → {f['arrive']} ·
{'non-stop' if not f['stops'] else '1 stop'} · <b>₹{f['fare_per_adult']:,}</b>
<form method="get" action="/skytrip/passengers"><input type="hidden" name="offer" value="{f['offer_id']}">
<button type="submit" aria-label="Select {html.escape(f['airline'])} {f['flight_no']} at {f['depart']} for ₹{f['fare_per_adult']:,}">Select</button></form></div>"""
        for f in flights
    )  # fmt: skip
    return _page(f"Flights {flights[0]['from']} → {flights[0]['to']} on {date}", rows, phone)


@router.get("/passengers", response_class=HTMLResponse)
async def passengers(request: Request, offer: str):
    phone, _ = _user(request)
    if not phone:
        return _login_redirect()
    f = mock.flights.get_offer(offer)
    if not f:
        return _page("Flight not found", "<a href='/skytrip/search'>Search again</a>", phone, 404)
    return _page("Passenger details", f"""<p>{html.escape(f['airline'])} {f['flight_no']} · {f['from']} → {f['to']} ·
{f['date']} {f['depart']}</p><form method="post" action="/skytrip/review"><input type="hidden" name="offer" value="{offer}">
<label for="name">Full name</label><input id="name" name="name" required>
<label for="age">Age</label><input id="age" name="age" inputmode="numeric" required>
<label for="gender">Gender</label><select id="gender" name="gender"><option value="M">Male</option>
<option value="F">Female</option><option value="O">Other</option></select>
<button type="submit">Continue to review</button></form>""", phone)


@router.post("/review", response_class=HTMLResponse)
async def review(request: Request, offer: str = Form(...), name: str = Form(...), age: str = Form(...),
                 gender: str = Form(...)):  # fmt: skip
    phone, sess = _user(request)
    if not phone:
        return _login_redirect()
    f = mock.flights.get_offer(offer)
    if not f:
        return _page("Flight not found", "<a href='/skytrip/search'>Search again</a>", phone, 404)
    sess["draft"] = {"offer": f, "name": name.strip(), "age": age.strip(), "gender": gender}
    return _page("Review your booking", f"""<div class="card"><b>{html.escape(f['airline'])} {f['flight_no']}</b><br>
{f['from']} → {f['to']} · {f['date']} · {f['depart']} → {f['arrive']}<br>
Passenger: {html.escape(name)} ({html.escape(age)}, {html.escape(gender)})<br><b>Total: ₹{f['fare_per_adult']:,}</b></div>
<form method="post" action="/skytrip/book"><button class="danger" type="submit">Book now and pay ₹{f['fare_per_adult']:,}</button></form>
<p class="muted">Demo site: no real payment is taken.</p>""", phone)


@router.post("/book", response_class=HTMLResponse)
async def book(request: Request):
    phone, sess = _user(request)
    if not phone:
        return _login_redirect()
    draft = sess.get("draft") if sess else None
    if not draft:
        return _page("Nothing to book", "<a href='/skytrip/search'>Search flights</a>", phone, 400)
    f = draft["offer"]
    pnr = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    trip = {"pnr": pnr, "flight": f"{f['airline']} {f['flight_no']}", "route": f"{f['from']} → {f['to']}",
            "when": f"{f['date']} {f['depart']}", "passenger": draft["name"], "amount": f["fare_per_adult"]}  # fmt: skip
    _trips.setdefault(phone, []).append(trip)
    sess["draft"] = None
    return _page("Booking confirmed", f"""<p class='ok'><b>Your flight is booked.</b></p><div class="card">
PNR: <b>{pnr}</b><br>{html.escape(trip['flight'])} · {trip['route']} · {trip['when']}<br>
Passenger: {html.escape(trip['passenger'])} · Paid ₹{trip['amount']:,}</div>""", phone)


@router.get("/trips", response_class=HTMLResponse)
async def trips(request: Request):
    phone, _ = _user(request)
    if not phone:
        return _login_redirect()
    items = _trips.get(phone) or []
    body = "".join(f"<div class='card'>PNR <b>{t['pnr']}</b> · {html.escape(t['flight'])} · {t['route']} · {t['when']}</div>"
                   for t in items) or "<p>No trips yet.</p>"  # fmt: skip
    return _page("My trips", body, phone)
