import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import admin, pay, webhook
from app.config import settings
from app.db import init_db
from app.services.messaging import get_client

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info(
        "starting %s | whatsapp=%s | agent=%s | checkout=%s",
        settings.app_name,
        "MOCK" if settings.mock_whatsapp else "LIVE",
        f"{settings.llm_provider}:{settings.llm_model or 'default'}" if settings.llm_active else "DISABLED (no LLM_API_KEY)",
        settings.public_url,
    )
    yield
    await get_client().aclose()


app = FastAPI(
    title=settings.app_name,
    description="Personal AI agent on WhatsApp: books flights and trains and orders food via a tool-using LLM.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(webhook.router)
app.include_router(pay.router)
app.include_router(admin.router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {
        "status": "ok",
        "whatsapp_mode": "mock" if settings.mock_whatsapp else "live",
        "agent": settings.llm_provider if settings.llm_active else "disabled",
    }
