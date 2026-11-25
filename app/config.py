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
    llm_provider: str = "gemini"  # gemini | groq | openrouter
    llm_api_key: str = ""
    llm_model: str = ""  # empty => provider default (see app/agent/llm.py)
    llm_fallback_model: str = ""  # used once when the main model is unavailable (429 / 5xx / timeout)
    llm_fallback_provider: str = ""  # optional: a DIFFERENT provider for the fallback (e.g. gemini)
    llm_fallback_api_key: str = ""  # key for the fallback provider (defaults to LLM_API_KEY)
    llm_max_tokens: int = 700  # WhatsApp replies are short; Groq free tier caps output at 1000 tokens/minute
    llm_base_url: str = ""  # optional override of the provider's OpenAI-compatible endpoint
    llm_enabled: bool = True
    agent_max_steps: int = 12  # max tool-use iterations per user message (browser tasks need a few more)
    history_messages: int = 30  # short-term conversation memory kept per user

    # --- Voice notes (speech-to-text, OpenAI-compatible /audio/transcriptions; Groq Whisper by default)
    stt_base_url: str = "https://api.groq.com/openai/v1"
    stt_api_key: str = ""  # defaults to the Groq key used for the LLM / fallback
    stt_model: str = "whisper-large-v3-turbo"

    # --- Voice replies (text-to-speech): when the user sends a voice note, answer with a voice note
    tts_enabled: bool = True
    tts_model: str = "gemini-3.8-flash-tts"  # Gemini TTS speaks Hindi, English and Hinglish
    tts_voice: str = "Kore"
    tts_api_key: str = ""  # defaults to the Gemini key (main or fallback)
    ffmpeg_path: str = "ffmpeg"  # converts speech to WhatsApp voice-note format (OGG/Opus)

    # --- Browser agent (Playwright). Off by default: needs `pip install -r requirements-browser.txt` +
    # `playwright install chromium` and ~1-2 GB RAM, so it's meant for running locally / on a bigger server.
    browser_enabled: bool = False
    browser_headless: bool = True
    # "*" = any PUBLIC website. Local/private addresses stay blocked; this app's own site (SkyTrip, at
    # PUBLIC_BASE_URL) is allowed on its exact host+port. Replace "*" with a list (e.g. "wikipedia.org,bbc.com")
    # to restrict the browser to those sites only; an entry also covers its subdomains.
    browser_allowed_hosts: str = "*"
    browser_state_dir: str = "./browser_sessions"  # saved logins (cookies) per user; keep private
    demo_site_enabled: bool = True  # SkyTrip demo website at /skytrip

    tavily_api_key: str = ""  # optional: best web search (tavily.com free tier); falls back to free sources

    # --- Guardrails / payments
    public_base_url: str = ""  # empty => RENDER_EXTERNAL_URL (set automatically on Render) or localhost
    render_external_url: str = ""  # where the checkout page is reachable (ngrok URL)
    max_order_amount: int = 200000  # INR; agent refuses to prepare anything above this
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
    def public_url(self) -> str:
        return (self.public_base_url or self.render_external_url or "http://localhost:8000").rstrip("/")

    @property
    def stt_key(self) -> str:
        if self.stt_api_key:
            return self.stt_api_key
        if self.llm_provider == "groq":
            return self.llm_api_key
        if self.llm_fallback_provider == "groq":
            return self.llm_fallback_api_key
        return ""

    @property
    def tts_key(self) -> str:
        if self.tts_api_key:
            return self.tts_api_key
        if self.llm_provider == "gemini":
            return self.llm_api_key
        if self.llm_fallback_provider == "gemini":
            return self.llm_fallback_api_key
        return ""

    @property
    def tts_active(self) -> bool:
        return self.tts_enabled and bool(self.tts_key)

    @property
    def llm_active(self) -> bool:
        return self.llm_enabled and bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
