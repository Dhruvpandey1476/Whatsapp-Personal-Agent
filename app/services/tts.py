"""Text-to-speech for voice replies: Gemini TTS (natural Hindi / English / Hinglish) -> ffmpeg -> OGG/Opus,
the format WhatsApp plays as a voice note."""

import asyncio
import base64
import re

import httpx

from app.config import settings

MAX_CHARS = 900  # keep voice replies short; long text is slow to synthesize and tiring to listen to


def speakable(text: str) -> str:
    """Strip what shouldn't be read aloud: URLs, WhatsApp formatting, emojis, bullet symbols."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_~`]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]", "", text)
    text = re.sub(r"^\s*[•\-]\s*", "", text, flags=re.M)
    text = text.replace("₹", "rupees ").replace("→", " to ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        text = cut[: max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), 200) + 1]
    return text


async def _gemini_speech(text: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as http:
        r = await http.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.tts_model}:generateContent",
            headers={"x-goog-api-key": settings.tts_key},
            json={"contents": [{"parts": [{"text": text}]}],
                  "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                      "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": settings.tts_voice}}}}},
        )  # fmt: skip
    r.raise_for_status()
    data = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    return base64.b64decode(data["data"]), data.get("mimeType", "audio/L16;rate=24000")


async def _to_ogg_opus(audio: bytes, mime: str) -> bytes:
    if "wav" in mime.lower():
        source = []  # WAV has a header: let ffmpeg detect the format
    else:  # raw 16-bit PCM, e.g. "audio/L16;codec=pcm;rate=24000"
        rate = re.search(r"rate=(\d+)", mime)
        source = ["-f", "s16le", "-ar", rate.group(1) if rate else "24000", "-ac", "1"]
    proc = await asyncio.create_subprocess_exec(
        settings.ffmpeg_path, "-hide_banner", "-loglevel", "error", *source, "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", "32k", "-ac", "1", "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    out, err = await proc.communicate(audio)
    if proc.returncode != 0 or not out:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[:200]}")
    return out


async def synthesize(text: str) -> bytes:
    """Reply text -> OGG/Opus voice note bytes. Raises on failure (caller falls back to text)."""
    clean = speakable(text)
    if not clean:
        raise ValueError("nothing to speak")
    audio, mime = await _gemini_speech(clean)
    return await _to_ogg_opus(audio, mime)
