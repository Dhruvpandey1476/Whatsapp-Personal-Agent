"""Speech-to-text for WhatsApp voice notes (Groq Whisper by default; any OpenAI-compatible
/audio/transcriptions endpoint works). WhatsApp voice notes are OGG/Opus, which Whisper accepts directly."""

import httpx

from app.config import settings

# Biases Whisper toward the words this assistant hears most (place names, Hinglish), improving accuracy.
PROMPT = ("Personal assistant on WhatsApp, English or Hinglish. Flights, trains, food: Delhi, Mumbai, Bengaluru, "
          "Goa, Dubai, IndiGo, Rajdhani, 3A, sleeper, biryani, masala dosa, cash on delivery, kal, parso.")  # fmt: skip
EXT = {"audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/aac": "aac", "audio/amr": "amr",
       "audio/wav": "wav", "audio/webm": "webm"}  # fmt: skip
MAX_BYTES = 20 * 1024 * 1024


async def transcribe(audio: bytes, mime: str = "audio/ogg") -> str:
    if len(audio) > MAX_BYTES:
        raise ValueError("Voice note is too long")
    mime = mime.split(";")[0].strip()
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as http:
        r = await http.post(
            f"{settings.stt_base_url.rstrip('/')}/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.stt_key}"},
            data={"model": settings.stt_model, "prompt": PROMPT, "response_format": "json", "temperature": "0"},
            files={"file": (f"voice.{EXT.get(mime, 'ogg')}", audio, mime)},
        )
    r.raise_for_status()
    return r.json().get("text", "")
