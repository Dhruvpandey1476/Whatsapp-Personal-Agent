"""Thin async wrapper around the WhatsApp Cloud API (Graph API /{phone-number-id}/messages).

In mock mode (no credentials configured) nothing leaves the process: payloads are recorded in
`self.sent` and a fake message id is returned, so the whole bot can be developed and tested offline.
"""

import asyncio
import logging
import uuid
from typing import Any

import httpx

from app.config import Settings, settings

log = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.facebook.com"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# WhatsApp hard limits for interactive messages
MAX_BUTTONS = 3
MAX_BUTTON_TITLE = 20
MAX_LIST_ROWS = 10
MAX_ROW_TITLE = 24
MAX_ROW_DESCRIPTION = 72


class WhatsAppAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: int | None = None):
        super().__init__(f"WhatsApp API error {status_code} (code={code}): {message}")
        self.status_code = status_code
        self.code = code
        self.message = message


class WhatsAppClient:
    def __init__(self, cfg: Settings = settings, http: httpx.AsyncClient | None = None):
        self.cfg = cfg
        self.mock = cfg.mock_whatsapp
        self.sent: list[dict[str, Any]] = []  # mock-mode outbox
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(15.0))

    @property
    def _messages_url(self) -> str:
        return f"{GRAPH_BASE}/{self.cfg.whatsapp_api_version}/{self.cfg.whatsapp_phone_number_id}/messages"

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, payload: dict[str, Any], max_attempts: int = 3) -> dict[str, Any]:
        if self.mock:
            self.sent.append(payload)
            log.info("[mock-whatsapp] -> %s", payload)
            return {"messages": [{"id": f"wamid.mock.{uuid.uuid4().hex}"}]}

        headers = {"Authorization": f"Bearer {self.cfg.whatsapp_access_token}"}
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self._http.post(self._messages_url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt == max_attempts:
                    raise WhatsAppAPIError(0, f"network error: {exc}") from exc
                await asyncio.sleep(2 ** (attempt - 1))
                continue

            if resp.status_code < 400:
                return resp.json()
            if resp.status_code in RETRYABLE_STATUS and attempt < max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))
                continue

            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            raise WhatsAppAPIError(resp.status_code, err.get("message", resp.text), err.get("code"))
        raise AssertionError("unreachable")

    @staticmethod
    def _message_id(result: dict[str, Any]) -> str | None:
        msgs = result.get("messages") or []
        return msgs[0].get("id") if msgs else None

    async def send(self, to: str, message: dict[str, Any]) -> str | None:
        """Send a message object (the part after `type`, e.g. {"type": "text", "text": {...}})."""
        clean = {k: v for k, v in message.items() if not k.startswith("_")}
        payload = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to, **clean}
        return self._message_id(await self._post(payload))

    async def download_media(self, media_id: str) -> tuple[bytes, str]:
        """Inbound media (voice notes, images): resolve the media id to a short-lived URL, then fetch the bytes.
        Both calls need the access token. Returns (content, mime_type)."""
        if self.mock:
            raise WhatsAppAPIError(0, "media download is not available in mock mode")
        headers = {"Authorization": f"Bearer {self.cfg.whatsapp_access_token}"}
        meta = await self._http.get(f"{GRAPH_BASE}/{self.cfg.whatsapp_api_version}/{media_id}", headers=headers)
        if meta.status_code >= 400:
            raise WhatsAppAPIError(meta.status_code, meta.text[:200])
        info = meta.json()
        media = await self._http.get(info["url"], headers=headers, follow_redirects=True)
        if media.status_code >= 400:
            raise WhatsAppAPIError(media.status_code, "media download failed")
        return media.content, info.get("mime_type", "audio/ogg")

    async def upload_media(self, content: bytes, mime: str = "audio/ogg", filename: str = "reply.ogg") -> str:
        """Upload outbound media (e.g. a voice reply) and return its media id for sending."""
        if self.mock:
            self.sent.append({"upload": filename, "bytes": len(content)})
            return f"media.mock.{uuid.uuid4().hex}"
        headers = {"Authorization": f"Bearer {self.cfg.whatsapp_access_token}"}
        url = f"{GRAPH_BASE}/{self.cfg.whatsapp_api_version}/{self.cfg.whatsapp_phone_number_id}/media"
        resp = await self._http.post(url, headers=headers, data={"messaging_product": "whatsapp", "type": mime},
                                     files={"file": (filename, content, mime)})  # fmt: skip
        if resp.status_code >= 400:
            raise WhatsAppAPIError(resp.status_code, resp.text[:200])
        return resp.json()["id"]

    async def mark_read(self, wa_message_id: str) -> None:
        try:
            await self._post({"messaging_product": "whatsapp", "status": "read", "message_id": wa_message_id})
        except WhatsAppAPIError as exc:  # read receipts are best effort
            log.warning("mark_read failed: %s", exc)


# ---------------------------------------------------------------- message builders


def text_message(body: str) -> dict[str, Any]:
    return {"type": "text", "text": {"body": body[:4096], "preview_url": False}}


def buttons_message(body: str, buttons: list[tuple[str, str]], footer: str | None = None) -> dict[str, Any]:
    """buttons: [(id, title)], max 3."""
    interactive: dict[str, Any] = {
        "type": "button",
        "body": {"text": body[:1024]},
        "action": {
            "buttons": [
                {"type": "reply", "reply": {"id": bid, "title": title[:MAX_BUTTON_TITLE]}}
                for bid, title in buttons[:MAX_BUTTONS]
            ]
        },
    }
    if footer:
        interactive["footer"] = {"text": footer[:60]}
    return {"type": "interactive", "interactive": interactive}


def list_message(
    body: str,
    button: str,
    rows: list[tuple[str, str, str]],
    section_title: str = "Options",
    header: str | None = None,
) -> dict[str, Any]:
    """rows: [(id, title, description)], max 10."""
    interactive: dict[str, Any] = {
        "type": "list",
        "body": {"text": body[:4096]},
        "action": {
            "button": button[:20],
            "sections": [
                {
                    "title": section_title[:24],
                    "rows": [
                        {"id": rid, "title": title[:MAX_ROW_TITLE], "description": desc[:MAX_ROW_DESCRIPTION]}
                        for rid, title, desc in rows[:MAX_LIST_ROWS]
                    ],
                }
            ],
        },
    }
    if header:
        interactive["header"] = {"type": "text", "text": header[:60]}
    return {"type": "interactive", "interactive": interactive}


def cta_url_message(body: str, display_text: str, url: str, footer: str | None = None) -> dict[str, Any]:
    """A message with a single URL button (used for payment links)."""
    interactive: dict[str, Any] = {
        "type": "cta_url",
        "body": {"text": body[:1024]},
        "action": {"name": "cta_url", "parameters": {"display_text": display_text[:20], "url": url}},
    }
    if footer:
        interactive["footer"] = {"text": footer[:60]}
    return {"type": "interactive", "interactive": interactive}


def image_message(media_id: str, caption: str = "") -> dict[str, Any]:
    image: dict[str, Any] = {"id": media_id}
    if caption:
        image["caption"] = caption[:1024]
    return {"type": "image", "image": image}


def audio_message(media_id: str) -> dict[str, Any]:
    """An uploaded OGG/Opus file shows in WhatsApp as a voice note."""
    return {"type": "audio", "audio": {"id": media_id}}


def template_message(name: str, language: str, components: list | None = None) -> dict[str, Any]:
    template: dict[str, Any] = {"name": name, "language": {"code": language}}
    if components:
        template["components"] = components
    return {"type": "template", "template": template}


def describe(message: dict[str, Any]) -> str:
    """Human-readable body for logging outbound messages in the DB."""
    t = message["type"]
    if t == "text":
        return message["text"]["body"]
    if t == "interactive":
        return message["interactive"]["body"]["text"]
    if t == "template":
        return f"[template:{message['template']['name']}]"
    if t == "image":
        return "[image] " + message["image"].get("caption", "")
    if t == "audio":
        return message.get("_transcript") or "[voice reply]"
    return f"[{t}]"
