import hashlib
import hmac
import json
import os
import tempfile
import time
import uuid

# Configure the app for tests *before* it is imported.
_tmpdir = tempfile.mkdtemp()
os.environ.update(
    DATABASE_URL=f"sqlite+aiosqlite:///{_tmpdir}/test.db",
    WHATSAPP_MOCK="true",
    WHATSAPP_APP_SECRET="test-secret",
    WHATSAPP_VERIFY_TOKEN="verify-me",
    API_KEY="test-key",
    LLM_ENABLED="true",
    ANTHROPIC_API_KEY="test",
    PUBLIC_BASE_URL="http://test",
    ENVIRONMENT="test",
)

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.agent.agent import set_llm  # noqa: E402
from app.models import Booking, Contact, Message, ToolTrace  # noqa: E402
from app.services.messaging import get_client  # noqa: E402

USER = "919876543210"


@pytest.fixture(scope="session", autouse=True)
async def _db():
    await init_db()


@pytest.fixture(autouse=True)
async def _clean():
    async with SessionLocal() as db:
        for model in (ToolTrace, Message, Booking, Contact):
            await db.execute(delete(model))
        await db.commit()
    get_client().sent.clear()
    yield
    set_llm(None)


@pytest.fixture
async def client():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()


def _envelope(message: dict, wa_id: str = USER, name: str = "Dhruv Pandey") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "15550000000", "phone_number_id": "123"},
                            "contacts": [{"wa_id": wa_id, "profile": {"name": name}}],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


def text_payload(text: str, wa_id: str = USER, wamid: str | None = None) -> dict:
    return _envelope(
        {
            "from": wa_id,
            "id": wamid or f"wamid.{uuid.uuid4().hex}",
            "timestamp": str(int(time.time())),
            "type": "text",
            "text": {"body": text},
        },
        wa_id,
    )


def reply_payload(reply_id: str, title: str = "x", kind: str = "list_reply", wa_id: str = USER) -> dict:
    return _envelope(
        {
            "from": wa_id,
            "id": f"wamid.{uuid.uuid4().hex}",
            "timestamp": str(int(time.time())),
            "type": "interactive",
            "interactive": {"type": kind, kind: {"id": reply_id, "title": title}},
        },
        wa_id,
    )


def status_payload(wamid: str, status: str) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {"statuses": [{"id": wamid, "status": status, "recipient_id": USER}]},
                    }
                ]
            }
        ],
    }


async def post_webhook(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    body = json.dumps(payload).encode()
    return await client.post(
        "/webhook", content=body, headers={"X-Hub-Signature-256": sign(body), "Content-Type": "application/json"}
    )


def outbound() -> list[dict]:
    """Messages the bot sent (excluding read receipts)."""
    return [p for p in get_client().sent if "to" in p]


def last_out() -> dict:
    return outbound()[-1]


def row_ids(message: dict) -> list[str]:
    inter = message["interactive"]
    if inter["type"] == "list":
        return [r["id"] for s in inter["action"]["sections"] for r in s["rows"]]
    return [b["reply"]["id"] for b in inter["action"]["buttons"]]
