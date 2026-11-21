from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "WhatsApp Personal Agent"
    environment: str = "development"
    log_level: str = "INFO"

    # --- WhatsApp Cloud API
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_business_account_id: str = ""
    whatsapp_app_secret: str = ""
    whatsapp_verify_token: str = "change-me-verify-token"
    whatsapp_api_version: str = "v25.0"
    whatsapp_mock: bool | None = None  # None => mock when no access token

    api_key: str = "dev-api-key"
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    # --- Agent
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    llm_enabled: bool = True
    agent_max_steps: int = 8  # max tool-use iterations per user message
    history_messages: int = 30  # short-term conversation memory kept per user

    # --- Guardrails / payments
    public_base_url: str = "http://localhost:8000"  # where the checkout page is reachable (ngrok URL)
    max_order_amount: int = 25000  # INR; agent refuses to prepare anything above this
    payment_link_minutes: int = 15

    timezone: str = "Asia/Kolkata"

    @property
    def mock_whatsapp(self) -> bool:
        if self.whatsapp_mock is not None:
            return self.whatsapp_mock
        return not (self.whatsapp_access_token and self.whatsapp_phone_number_id)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def llm_active(self) -> bool:
        return self.llm_enabled and bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
