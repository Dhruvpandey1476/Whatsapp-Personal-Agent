"""Browser tools for the agent (enabled with BROWSER_ENABLED=true). Every action returns the new page state, so
the model sees the result right away without spending an extra step on reading the page."""

import logging

from app.agent.tools import ToolContext, ToolError
from app.browser.manager import BrowserError, get_browser
from app.config import settings
from app.services.messaging import get_client
from app.whatsapp.client import buttons_message, image_message

log = logging.getLogger(__name__)

BROWSER_TOOLS: list[dict] = [
    {
        "name": "browser_open",
        "description": "Open a web page in your real browser (only allowed sites). Returns the page text and its "
        "numbered interactive elements. Saved logins are reused automatically.",
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
    {
        "name": "browser_fill",
        "description": "Type into one or more fields of the current page (inputs, selects) using element ids from "
        "the LATEST page state. Never use it for passwords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"element_id": {"type": "integer"}, "value": {"type": "string"}},
                        "required": ["element_id", "value"],
                    },
                }
            },
            "required": ["fields"],
        },
    },
    {
        "name": "browser_click",
        "description": "Click a button or link by element id from the LATEST page state. Final actions (book, pay, "
        "confirm, place order) are not clicked directly: the user gets a screenshot with Approve/Cancel buttons, "
        "and the click happens only if they approve.",
        "input_schema": {"type": "object", "properties": {"element_id": {"type": "integer"}}, "required": ["element_id"]},
    },
    {
        "name": "browser_read",
        "description": "Re-read the current page (text and element ids) without doing anything. Big pages list "
        "at most 80 controls; pass `find` (e.g. 'Buy now', 'Add to cart', 'Use Email') to list every control whose "
        "label contains that text, wherever it is on the page.",
        "input_schema": {"type": "object", "properties": {"find": {"type": "string"}}},
    },
    {
        "name": "browser_screenshot",
        "description": "Send the user a screenshot of the current page on WhatsApp.",
        "input_schema": {"type": "object", "properties": {"caption": {"type": "string"}}},
    },
    {
        "name": "browser_forget_logins",
        "description": "Log the user out of all websites and delete saved browser logins. Only when the user asks.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def _send_screenshot(ctx: ToolContext, caption: str) -> None:
    png = await get_browser().screenshot(ctx.contact.wa_id)
    media_id = await get_client().upload_media(png, "image/png", "page.png")
    ctx.outbox.append(image_message(media_id, caption[:1000]))


async def _run(coro):
    try:
        return await coro
    except BrowserError as exc:
        raise ToolError(str(exc)) from exc
    except ImportError as exc:
        raise ToolError("Browser support isn't installed on this server.") from exc
    except Exception as exc:  # noqa: BLE001 - timeouts, missing elements: let the model recover
        log.warning("browser action failed: %s", exc)
        raise ToolError(f"Browser action failed: {str(exc).splitlines()[0][:200]}") from exc


async def browser_open(ctx: ToolContext, url: str) -> dict:
    return await _run(get_browser().open(ctx.contact.wa_id, url))


async def browser_fill(ctx: ToolContext, fields: list[dict]) -> dict:
    return await _run(get_browser().fill(ctx.contact.wa_id, fields))


async def browser_read(ctx: ToolContext, find: str = "") -> dict:
    return await _run(get_browser().snapshot(ctx.contact.wa_id, find))


async def browser_click(ctx: ToolContext, element_id: int) -> dict:
    result = await _run(get_browser().click(ctx.contact.wa_id, element_id))
    if "approval_token" not in result:
        return result
    token, label = result["approval_token"], result["label"]
    try:
        await _send_screenshot(ctx, f"About to click “{label}” on {result['url']}")
    except Exception:  # noqa: BLE001 - the approval still works without the picture
        log.exception("screenshot for approval failed")
    ctx.show("confirm", buttons_message(
        f"🛡️ I'm about to click *{label}* on the website. This is a final action.\nShall I go ahead?",
        [(f"br:approve:{token}", "✅ Approve"), (f"br:cancel:{token}", "❌ Cancel")],
        footer="Nothing happens unless you approve"))  # fmt: skip
    return {"status": "awaiting_user_approval", "action": label,
            "note": "A screenshot and Approve/Cancel buttons will appear right AFTER your message. It has NOT been "
                    "clicked. Briefly say what you're about to do and ask the user to tap Approve on the message "
                    "below; don't claim it's done."}  # fmt: skip


async def browser_screenshot(ctx: ToolContext, caption: str = "") -> dict:
    await _run(_send_screenshot(ctx, caption or "Here's the page"))
    return {"sent": True}


async def browser_forget_logins(ctx: ToolContext) -> dict:
    await _run(get_browser().forget(ctx.contact.wa_id))
    return {"done": True, "note": "Logged out of all sites; saved logins deleted."}


BROWSER_REGISTRY = {
    "browser_open": browser_open,
    "browser_fill": browser_fill,
    "browser_click": browser_click,
    "browser_read": browser_read,
    "browser_screenshot": browser_screenshot,
    "browser_forget_logins": browser_forget_logins,
}


def browser_prompt() -> str:
    site = f"{settings.public_url}/skytrip"
    return f"""

Browser (you have a real web browser via browser_* tools):
- You can operate these websites for the user: SkyTrip, an airline booking site, at {site} (log in by OTP).
  Use it when the user asks you to book or check something *on the website / SkyTrip*, or to log in for them.
- You can also open any public website the user asks for (news, Wikipedia, travel, shops, government sites...)
  to read, search or fill forms, e.g. https://en.wikipedia.org/wiki/Special:Search?search=Rajdhani+Express.
- When the user asks you to log in to, book on, open, go to or check a SPECIFIC website, use the browser right
  away (browser_open https://www.<site>), not web_search. Fix obvious typos in site names yourself
  (makemytip.com -> makemytrip.com) and say which site you opened. web_search is for general questions.
- On sites other than SkyTrip: log in only via an OTP the user sends you (never passwords). If a site shows a
  CAPTCHA or blocks bots, tell the user; never try to get around it. Don't edit or post content unless asked;
  final actions always go through the user's approval.
- Login by OTP: open the site, type the user's mobile number (their WhatsApp number is {{wa_id}}), click
  "Send code", then ask the user to send you the 6-digit code they just received. Never guess or invent a code,
  and never type passwords. Once logged in, the login is remembered.
- Always use element ids from the most recent page state. Fill all fields of a form in one browser_fill.
- Final actions (book, pay, confirm): when the user has asked you to book and everything is filled in, call
  browser_click on that button right away (don't ask "shall I book?" and don't take a screenshot first). The tool
  itself sends the user a screenshot plus Approve/Cancel buttons and returns status awaiting_user_approval.
  Only mention the Approve button after that status; then stop and wait. Never say it's done before a system
  note says it was approved and clicked.
- Summarise what you see for the user in a short message; don't paste page text.
- If the user asked you to use a website (e.g. SkyTrip) and it fails, say what went wrong and retry or ask them;
  don't silently switch to the normal flight tools.
- Never say "let me do X" and then stop: either do it now with a tool call, or ask the user.
- If a button or field you need isn't in the element list, call browser_read with find="<its text>" before giving
  up (big pages only list 80 controls). If it's still not there, say exactly what you couldn't find; never invent
  reasons like "regional restrictions" or "login limitations"."""
