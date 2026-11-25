"""Browser agent: the SkyTrip demo site, the approval gate, and a real headless-Chromium run end to end."""

import asyncio
import re
import socket
from datetime import date, timedelta

import pytest

from app.agent.agent import set_llm
from app.browser import manager as bm
from app.services.messaging import get_client
from tests.conftest import USER, outbound, post_webhook, reply_payload, row_ids, text_payload
from tests.test_agent import ScriptedLLM, body, say, tool

D = (date.today() + timedelta(days=5)).isoformat()


def last_otp() -> str:
    texts = [p["text"]["body"] for p in get_client().sent if p.get("type") == "text"]
    return re.search(r"login code: (\d{6})", [t for t in texts if "SkyTrip login code" in t][-1]).group(1)


# ---------------------------------------------------------------- demo site (plain HTTP)


async def test_skytrip_otp_login_search_and_book(client):
    await post_webhook(client, text_payload("hi"))  # the number has chatted with the assistant
    r = await client.post("/skytrip/login/send", data={"phone": USER})
    assert "We sent a 6-digit code" in r.text
    code = last_otp()

    bad = await client.post("/skytrip/login/verify", data={"phone": USER, "otp": "000000" if code != "000000" else "111111"})
    assert bad.status_code == 400 and "Wrong code" in bad.text
    ok = await client.post("/skytrip/login/verify", data={"phone": USER, "otp": code}, follow_redirects=False)
    assert ok.status_code == 303 and "skytrip_session" in ok.headers["set-cookie"]
    client.cookies.set("skytrip_session", ok.cookies["skytrip_session"])

    res = await client.get("/skytrip/results", params={"origin": "Delhi", "destination": "Mumbai", "date": D})
    offer = re.search(r'name="offer" value="([^"]+)"', res.text).group(1)
    review = await client.post("/skytrip/review", data={"offer": offer, "name": "Dhruv Pandey", "age": "24", "gender": "M"})
    assert "Book now and pay" in review.text
    booked = await client.post("/skytrip/book")
    assert "Your flight is booked" in booked.text and re.search(r"PNR: <b>[A-Z0-9]{6}</b>", booked.text)


async def test_skytrip_requires_login_and_unknown_numbers_get_no_code(client):
    r = await client.get("/skytrip/search", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/skytrip/login"
    r = await client.post("/skytrip/login/send", data={"phone": "919000000001"})
    assert "send a code to that number" in r.text


# ---------------------------------------------------------------- safety rules


def test_final_actions_are_detected_and_hosts_restricted(monkeypatch):
    assert bm.is_sensitive("Book now and pay ₹3,850") and bm.is_sensitive("Place order") and bm.is_sensitive("Pay")
    assert not any(bm.is_sensitive(x) for x in ("Search flights", "Send code", "Verify and log in", "Select", "Facebook"))
    monkeypatch.setattr(bm.settings, "browser_allowed_hosts", "localhost")
    assert bm.allowed("http://localhost:8000/skytrip") and not bm.allowed("https://www.irctc.co.in/")
    monkeypatch.setattr(bm.settings, "browser_allowed_hosts", "localhost,wikipedia.org")
    assert bm.allowed("https://en.wikipedia.org/wiki/Rajdhani_Express") and bm.allowed("https://hi.wikipedia.org/")
    assert not bm.allowed("https://wikipedia.org.evil.com/") and not bm.allowed("https://notwikipedia.org/")
    assert bm.is_sensitive("Edit") and bm.is_sensitive("Publish changes")  # edits need approval
    monkeypatch.setattr(bm, "_is_public_host",
                        lambda h: h not in ("192.168.1.1", "169.254.169.254", "internal.corp", "localhost", "127.0.0.1"))
    monkeypatch.setattr(bm.settings, "browser_allowed_hosts", "*")
    monkeypatch.setattr(bm.settings, "public_base_url", "http://localhost:8000")
    assert bm.allowed("https://www.bbc.com/news") and bm.allowed("https://www.irctc.co.in/")  # any public site
    assert bm.allowed("http://localhost:8000/skytrip")  # this app's own site, exact host + port
    assert not bm.allowed("http://localhost:5432/") and not bm.allowed("http://127.0.0.1:8000/")  # other local services
    assert not bm.allowed("http://192.168.1.1/admin") and not bm.allowed("http://169.254.169.254/latest/meta-data")
    assert not bm.allowed("file:///C:/Windows/win.ini") and not bm.allowed("javascript:alert(1)")


class FakeBrowser:
    def __init__(self):
        self.approved = []

    async def click(self, wa_id, element_id):
        return {"approval_token": "TOK", "label": "Book now and pay ₹3,850", "url": "http://localhost/skytrip/review"}

    async def screenshot(self, wa_id):
        return b"\x89PNG"

    def pending(self, wa_id, token):
        return {"label": "Book now and pay ₹3,850", "url": "http://localhost/skytrip/review"} if token == "TOK" else None

    def cancel(self, wa_id, token):
        return self.pending(wa_id, token)

    async def approve(self, wa_id, token):
        self.approved.append(token)
        return {"title": "Booking confirmed · SkyTrip", "text": "Your flight is booked. PNR: ABC123", "elements": []}


async def test_final_click_waits_for_users_approval(client, monkeypatch):
    monkeypatch.setattr(bm.settings, "browser_enabled", True)
    fake = FakeBrowser()
    bm.set_browser(fake)
    try:
        set_llm(ScriptedLLM(tool("browser_click", element_id=7), say("Please approve the booking.")))
        await post_webhook(client, text_payload("book it on skytrip"))
        assert fake.approved == []  # the model asked to click "Book", but nothing was clicked
        assert row_ids(outbound()[-1]) == ["br:approve:TOK", "br:cancel:TOK"]
        assert any(m["type"] == "image" for m in outbound())  # screenshot of what will be clicked

        await post_webhook(client, reply_payload("br:approve:TOK", kind="button_reply"))
        assert fake.approved == ["TOK"] and "PNR: ABC123" in body(outbound()[-1])
        await post_webhook(client, reply_payload("br:approve:NOPE", kind="button_reply"))
        assert "expired" in body(outbound()[-1])
    finally:
        bm.set_browser(None)


# ---------------------------------------------------------------- real browser, real site, end to end


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_real_browser_logs_in_with_otp_and_books_after_approval(client, monkeypatch, tmp_path):
    pytest.importorskip("playwright")
    import uvicorn

    from app.main import app

    await post_webhook(client, text_payload("hi"))  # register the user so the site can deliver the OTP
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(bm.settings, "browser_allowed_hosts", "127.0.0.1")
    monkeypatch.setattr(bm.settings, "browser_state_dir", str(tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    serve = asyncio.create_task(server.serve())
    browser = bm.BrowserManager()
    try:
        while not server.started:
            await asyncio.sleep(0.05)

        def eid(state, pattern):
            return next(e["id"] for e in state["elements"] if re.search(pattern, e["label"], re.I))

        s = await browser.open(USER, f"{base}/skytrip")
        assert "Log in" in s["title"]
        s = await browser.fill(USER, [{"element_id": eid(s, "mobile"), "value": USER}])
        s = await browser.click(USER, eid(s, "send code"))
        s = await browser.fill(USER, [{"element_id": eid(s, "one-time code"), "value": last_otp()}])
        s = await browser.click(USER, eid(s, "verify"))
        assert "Search flights" in s["title"]

        s = await browser.fill(USER, [{"element_id": eid(s, "^from"), "value": "Delhi"},
                                      {"element_id": eid(s, "^to$"), "value": "Mumbai"},
                                      {"element_id": eid(s, "^date"), "value": D}])  # fmt: skip
        s = await browser.click(USER, eid(s, "^search flights"))
        s = await browser.click(USER, eid(s, "^select "))
        s = await browser.fill(USER, [{"element_id": eid(s, "full name"), "value": "Dhruv Pandey"},
                                      {"element_id": eid(s, "^age"), "value": "24"},
                                      {"element_id": eid(s, "gender"), "value": "M"}])  # fmt: skip
        s = await browser.click(USER, eid(s, "continue"))
        book = next(e for e in s["elements"] if e.get("needs_user_approval"))

        pending = await browser.click(USER, book["id"])  # final action: NOT clicked, approval token instead
        assert "approval_token" in pending and "Review" in (await browser.snapshot(USER))["title"]
        done = await browser.approve(USER, pending["approval_token"])
        assert "Booking confirmed" in done["title"] and re.search(r"PNR: [A-Z0-9]{6}", done["text"])
        assert (tmp_path / f"{USER}.json").exists()  # login saved for next time
    finally:
        await browser.close()
        server.should_exit = True
        await serve


async def test_without_browser_the_agent_is_told_to_say_so(client):
    llm = ScriptedLLM(say("The browser agent only runs in the local demo."))
    set_llm(llm)
    await post_webhook(client, text_payload("login to skytrip and book a flight"))
    system = llm.calls[0]["system"]
    assert "No browser on this server" in system and "browser_open" not in system


async def test_agent_is_told_the_real_browser_state_each_turn(client, monkeypatch):
    """After a restart the chat history still says 'login page open, code sent'. The agent must be told the truth."""
    monkeypatch.setattr(bm.settings, "browser_enabled", True)
    bm.set_browser(bm.BrowserManager())  # fresh manager: nothing open, no browser process started
    try:
        llm = ScriptedLLM(say("I'll start again and request a new code."))
        set_llm(llm)
        await post_webhook(client, text_payload("985581"))
        system = llm.calls[0]["system"]
        assert "the user's browser showed: nothing is open" in system and "request a NEW code" in system
    finally:
        bm.set_browser(None)
