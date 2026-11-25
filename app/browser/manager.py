"""Real browser for the agent (Playwright, headless Chromium).

- One browser process; one isolated context per WhatsApp user, so logins never mix between users.
- Logins persist: cookies are saved per user to disk (BROWSER_STATE_DIR) and restored, so a site needs one OTP.
- The page is given to the LLM as compact text plus numbered interactive elements ([3] button "Send code").
- Safety, enforced here in code (not left to the LLM):
  * any public website ("*"), but never local / private-network addresses except the listed demo hosts;
  * password fields are never filled (log in with OTP instead);
  * clicks on final actions (book / pay / confirm / place order...) are NOT performed: they become a pending
    approval that only the user's WhatsApp "Approve" tap can execute.
"""

import asyncio
import ipaddress
import logging
import re
import secrets
import socket
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings

log = logging.getLogger(__name__)

SENSITIVE = re.compile(r"\b(pay|book|place\s+order|buy|purchase|checkout|confirm(\s+(booking|order|payment))?|"
                       r"submit\s+order|transfer|delete\s+account|edit|publish|save\s+changes)\b", re.I)  # fmt: skip

SNAPSHOT_JS = """
() => {
  document.querySelectorAll('[data-agent-id]').forEach(e => e.removeAttribute('data-agent-id'));
  // Real controls, plus things that only *behave* clickable (many sites build links as <div>/<span> with a click
  // handler, e.g. Flipkart's "Use Email-ID"): ARIA roles, onclick, keyboard-focusable, or a pointer cursor.
  const CONTROLS = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=tab],' +
                   '[role=menuitem],[role=option],[role=checkbox],[role=radio],[role=switch],[onclick],[tabindex]';
  const candidates = new Set(document.querySelectorAll(CONTROLS));
  for (const el of document.querySelectorAll('div,span,li,p,label,img,svg')) {
    if (candidates.size > 1500) break;
    const t = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('alt') || '').trim();
    const handler = typeof el.onclick === 'function';  // click handler set from JavaScript (no HTML attribute)
    if (!t || t.length > 60 || (!handler && getComputedStyle(el).cursor !== 'pointer')) continue;
    // keep the outermost pointer element of a group (not each nested span), and skip ones inside a real control
    const parent = el.parentElement;
    if (parent && getComputedStyle(parent).cursor === 'pointer' && (parent.innerText || '').trim() === t) continue;
    if (el.closest('a[href],button,[role=button],[role=link]')) continue;
    candidates.add(el);
  }
  const els = [...candidates].sort((a, b) =>
    a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);  // page order
  const out = []; let n = 0;
  for (const el of els) {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (el.type === 'hidden' || st.visibility === 'hidden' || st.display === 'none' || (!r.width && !r.height)) continue;
    if (el.getAttribute('tabindex') === '-1' && !el.matches('a[href],button,input,select,textarea')) continue;
    n++; el.setAttribute('data-agent-id', String(n));
    let label = el.getAttribute('aria-label') || '';
    if (!label && el.labels && el.labels.length) label = el.labels[0].innerText;
    if (!label) label = el.innerText || el.value || el.placeholder || el.name || '';
    const item = {id: n, tag: el.tagName.toLowerCase(), label: label.trim().replace(/\\s+/g, ' ').slice(0, 90)};
    if (el.tagName === 'INPUT') { item.type = el.type; item.value = el.type === 'password' ? '' : el.value; }
    if (el.tagName === 'TEXTAREA') item.value = el.value;
    if (el.placeholder) item.placeholder = el.placeholder;
    if (el.tagName === 'SELECT') item.options = [...el.options].slice(0, 12).map(o => o.value + '=' + o.text);
    out.push(item);
    if (n >= 60) break;
  }
  const main = document.querySelector('main, [role=main]') || document.body;  // skip site menus and footers
  const text = (main ? main.innerText : '').replace(/\\n{3,}/g, '\\n\\n').trim();
  return {title: document.title, url: location.href, text: text.slice(0, 2500), elements: out};
}
"""


class BrowserError(Exception):
    """User-facing browser problem, returned to the LLM so it can recover."""


def _is_public_host(host: str) -> bool:
    """True if every address the host resolves to is on the public internet (not localhost, LAN, cloud metadata)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def allowed(url: str) -> bool:
    """Which sites the agent's browser may open.

    BROWSER_ALLOWED_HOSTS="*" (default): any PUBLIC website, plus the explicitly listed local hosts (the SkyTrip
    demo). Private/internal addresses stay blocked so a web page can't use the agent to reach the local network.
    A comma list without "*" restricts the browser to those sites (and their subdomains) only."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        return False
    hosts = {h.strip().lower() for h in settings.browser_allowed_hosts.split(",") if h.strip()}
    own = urlparse(settings.public_url)
    default_port = {"http": 80, "https": 443}
    # This app's own site (SkyTrip demo) is allowed on its exact host AND port only, so "localhost" can't be used
    # to reach other local services (databases, admin panels) on other ports.
    if host == (own.hostname or "").lower() and (parsed.port or default_port[parsed.scheme]) == (
        own.port or default_port.get(own.scheme, 80)
    ):
        return True
    if any(host == h or host.endswith("." + h) for h in hosts if h != "*"):
        return True
    return "*" in hosts and _is_public_host(host)


def is_sensitive(label: str) -> bool:
    return bool(SENSITIVE.search(label or ""))


class BrowserManager:
    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._contexts: dict[str, object] = {}
        self._pages: dict[str, object] = {}
        self._elements: dict[str, dict[int, dict]] = {}  # wa_id -> last snapshot's elements by id
        self._pending: dict[str, dict[str, dict]] = {}  # wa_id -> token -> {"label", "tag", "url"}
        self._lock = asyncio.Lock()
        self.state_dir = Path(settings.browser_state_dir)

    # ------------------------------------------------------------------ lifecycle
    async def _ensure_browser(self):
        async with self._lock:
            if self._browser is None:
                from playwright.async_api import async_playwright  # optional dependency

                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=settings.browser_headless)
        return self._browser

    def _state_file(self, wa_id: str) -> Path:
        return self.state_dir / f"{re.sub(r'[^0-9]', '', wa_id)}.json"

    async def _page(self, wa_id: str):
        page = self._pages.get(wa_id)
        if page is not None and not page.is_closed():
            return page
        browser = await self._ensure_browser()
        state = self._state_file(wa_id)
        ctx = await browser.new_context(
            storage_state=str(state) if state.exists() else None,
            viewport={"width": 1100, "height": 800}, locale="en-IN",
        )  # fmt: skip
        ctx.set_default_timeout(10_000)
        self._contexts[wa_id] = ctx
        self._pages[wa_id] = page = await ctx.new_page()
        return page

    async def _save(self, wa_id: str) -> None:
        ctx = self._contexts.get(wa_id)
        if ctx is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            await ctx.storage_state(path=str(self._state_file(wa_id)))

    async def close(self) -> None:
        for ctx in list(self._contexts.values()):
            await ctx.close()
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._contexts.clear(), self._pages.clear()
        self._browser = self._pw = None

    async def forget(self, wa_id: str) -> None:
        """Log out everywhere: close the user's context and delete saved cookies."""
        ctx = self._contexts.pop(wa_id, None)
        self._pages.pop(wa_id, None)
        self._pending.pop(wa_id, None)
        if ctx is not None:
            await ctx.close()
        self._state_file(wa_id).unlink(missing_ok=True)

    # ------------------------------------------------------------------ reading
    async def _settle(self, page) -> None:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=8000)
            await page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:  # noqa: BLE001 - pages that keep connections open never go idle
            pass

    async def snapshot(self, wa_id: str) -> dict:
        page = await self._page(wa_id)
        if page.url not in ("about:blank", "") and not allowed(page.url):
            await page.goto("about:blank")
            raise BrowserError("The page left the allowed sites, so I stopped there.")
        state = await page.evaluate(SNAPSHOT_JS)
        self._elements[wa_id] = {e["id"]: e for e in state["elements"]}
        for e in state["elements"]:
            if e["tag"] in ("button", "a", "input") and is_sensitive(e["label"]):
                e["needs_user_approval"] = True
        return state

    async def describe(self, wa_id: str) -> str:
        """What the user's browser shows right now, without starting a browser. Told to the agent every turn,
        because the conversation history can't know about restarts or expired pages."""
        page = self._pages.get(wa_id)
        if page is None or page.is_closed() or page.url in ("", "about:blank"):
            return ("nothing is open (fresh browser: any earlier page or login code from this chat is gone, "
                    "so start again with browser_open; request a NEW code if you need to log in)")
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001
            title = ""
        return f"'{title}' at {page.url}"

    async def screenshot(self, wa_id: str) -> bytes:
        return await (await self._page(wa_id)).screenshot(type="png")

    # ------------------------------------------------------------------ actions
    async def open(self, wa_id: str, url: str) -> dict:
        if not allowed(url):
            raise BrowserError("That address isn't allowed: only public websites (and the SkyTrip demo) can be opened, "
                               "never local or private-network addresses.")
        page = await self._page(wa_id)
        await page.goto(url)
        await self._settle(page)
        await self._save(wa_id)
        return await self.snapshot(wa_id)

    def _element(self, wa_id: str, element_id: int) -> dict:
        el = self._elements.get(wa_id, {}).get(int(element_id))
        if el is None:
            raise BrowserError(f"Element {element_id} isn't on the current page. Use ids from the latest page state.")
        return el

    async def fill(self, wa_id: str, fields: list[dict]) -> dict:
        page = await self._page(wa_id)
        for f in fields:
            el = self._element(wa_id, f["element_id"])
            if el.get("type") == "password":
                raise BrowserError("I never type passwords. Use the site's OTP login instead.")
            loc = page.locator(f'[data-agent-id="{int(f["element_id"])}"]')
            if el["tag"] == "select":
                value = str(f["value"])
                try:
                    await loc.select_option(value=value)
                except Exception:  # noqa: BLE001
                    await loc.select_option(label=value)
            else:
                await loc.fill(str(f["value"]))
        return await self.snapshot(wa_id)

    async def click(self, wa_id: str, element_id: int) -> dict:
        """Click, unless it's a final action: then store it for the user's approval and return the token."""
        el = self._element(wa_id, element_id)
        page = await self._page(wa_id)
        if is_sensitive(el["label"]):
            token = secrets.token_urlsafe(8)
            self._pending.setdefault(wa_id, {})[token] = {"label": el["label"], "tag": el["tag"], "url": page.url}
            return {"approval_token": token, "label": el["label"], "url": page.url}
        await page.locator(f'[data-agent-id="{int(element_id)}"]').click()
        await self._settle(page)
        await self._save(wa_id)
        return await self.snapshot(wa_id)

    def pending(self, wa_id: str, token: str) -> dict | None:
        return self._pending.get(wa_id, {}).get(token)

    def cancel(self, wa_id: str, token: str) -> dict | None:
        return self._pending.get(wa_id, {}).pop(token, None)

    async def approve(self, wa_id: str, token: str) -> dict:
        """Perform a final action the user approved on WhatsApp. Code path only: the LLM can't call this."""
        item = self._pending.get(wa_id, {}).pop(token, None)
        if item is None:
            raise BrowserError("That approval has expired or was already used.")
        page = await self._page(wa_id)
        if page.url != item["url"]:
            raise BrowserError("The page changed since you were asked, so I didn't click anything.")
        await self.snapshot(wa_id)  # re-tag elements on the current page, then find the same button again
        match = next((e for e in self._elements[wa_id].values()
                      if e["label"] == item["label"] and e["tag"] == item["tag"]), None)  # fmt: skip
        if match is None:
            raise BrowserError(f"I couldn't find '{item['label']}' on the page any more.")
        await page.locator(f'[data-agent-id="{match["id"]}"]').click()
        await self._settle(page)
        await self._save(wa_id)
        return await self.snapshot(wa_id)


_manager: BrowserManager | None = None


def get_browser() -> BrowserManager:
    global _manager
    if _manager is None:
        _manager = BrowserManager()
    return _manager


def set_browser(manager: BrowserManager | None) -> None:
    global _manager
    _manager = manager
