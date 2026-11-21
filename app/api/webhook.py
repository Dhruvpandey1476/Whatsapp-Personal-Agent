import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response, status

from app.config import settings
from app.security import verify_meta_signature
from app.services.inbound import process_webhook

log = logging.getLogger(__name__)
router = APIRouter(tags=["webhook"])


@router.get("/webhook", summary="Meta webhook verification handshake")
async def verify_webhook(
    mode: str | None = Query(None, alias="hub.mode"),
    token: str | None = Query(None, alias="hub.verify_token"),
    challenge: str | None = Query(None, alias="hub.challenge"),
) -> Response:
    if mode == "subscribe" and token == settings.whatsapp_verify_token and challenge:
        return Response(content=challenge, media_type="text/plain")
    raise HTTPException(status.HTTP_403_FORBIDDEN, "Verification failed")


@router.post("/webhook", summary="Receive WhatsApp messages and status updates")
async def receive_webhook(request: Request, background: BackgroundTasks) -> dict:
    raw = await request.body()

    if settings.whatsapp_app_secret:
        if not verify_meta_signature(raw, request.headers.get("X-Hub-Signature-256"), settings.whatsapp_app_secret):
            log.warning("rejected webhook with invalid signature")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid signature")
    elif settings.environment == "production":
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "WHATSAPP_APP_SECRET not configured")

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid JSON") from exc

    # ACK immediately; Meta expects a fast 200 and retries otherwise. Processing (DB, LLM, replies)
    # happens after the response is sent.
    background.add_task(process_webhook, payload)
    return {"status": "received"}
