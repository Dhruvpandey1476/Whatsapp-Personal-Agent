"""Terminal WhatsApp simulator - chat with the running bot without a phone or Meta account.

Sends correctly signed webhook payloads (exactly what Meta would send) to the server and renders the
bot's replies by reading the conversation from the internal API. Type a number to tap a button/row.

    python scripts/simulate.py                       # talks to http://localhost:8000
    python scripts/simulate.py --phone 919999999999 --name "Asha"
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import uuid

import httpx

sys.stdout.reconfigure(encoding="utf-8")  # emoji on Windows terminals


def load_dotenv() -> None:
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def envelope(phone: str, name: str, message: dict) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "SIM", "changes": [{"field": "messages", "value": {
            "messaging_product": "whatsapp",
            "metadata": {"display_phone_number": "15550000000", "phone_number_id": "SIM"},
            "contacts": [{"wa_id": phone, "profile": {"name": name}}],
            "messages": [{"from": phone, "id": f"wamid.sim.{uuid.uuid4().hex}",
                          "timestamp": str(int(time.time())), **message}],
        }}]}],
    }  # fmt: skip


def render(msg: dict) -> tuple[list[tuple[str, str]], str]:
    """Print an outbound message; return the tappable (id, title) options and the reply kind."""
    payload = msg.get("payload") or {}
    options: list[tuple[str, str]] = []
    kind = "button_reply"
    if payload.get("type") == "interactive":
        inter = payload["interactive"]
        print(f"\n🤖 {inter['body']['text']}")
        if inter["type"] == "cta_url":
            p = inter["action"]["parameters"]
            print(f"   🔗 [{p['display_text']}] {p['url']}")
        elif inter["type"] == "button":
            options = [(b["reply"]["id"], b["reply"]["title"]) for b in inter["action"]["buttons"]]
        else:
            kind = "list_reply"
            print(f"   [{inter['action']['button']}]")
            for sec in inter["action"]["sections"]:
                for r in sec["rows"]:
                    options.append((r["id"], r["title"] + (f" - {r['description']}" if r.get("description") else "")))
        for i, (_, title) in enumerate(options, 1):
            print(f"   {i}. {title}")
    else:
        print(f"\n🤖 {msg['body']}")
    return options, kind


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--phone", default="919876543210")
    ap.add_argument("--name", default="Test User")
    args = ap.parse_args()

    secret = os.environ.get("WHATSAPP_APP_SECRET", "")
    api_key = os.environ.get("API_KEY", "dev-api-key")
    http = httpx.Client(base_url=args.url, timeout=60)
    seen: set[int] = set()
    options: list[tuple[str, str]] = []
    reply_kind = "button_reply"

    def history() -> list[dict]:
        r = http.get(f"/api/contacts/{args.phone}/messages", headers={"X-API-Key": api_key})
        return r.json() if r.status_code == 200 else []

    seen.update(m["id"] for m in history())
    print(f"📱 Chatting as {args.name} (+{args.phone}). Type a message, a number to tap an option, or /quit.\n")

    while True:
        try:
            text = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break

        if text.isdigit() and 1 <= int(text) <= len(options):
            rid, title = options[int(text) - 1]
            reply = {"id": rid, "title": title[:24]}
            message = {"type": "interactive", "interactive": {"type": reply_kind, reply_kind: reply}}
        else:
            message = {"type": "text", "text": {"body": text}}

        body = json.dumps(envelope(args.phone, args.name, message)).encode()
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        resp = http.post("/webhook", content=body, headers=headers)
        if resp.status_code != 200:
            print(f"❌ webhook rejected: {resp.status_code} {resp.text}")
            continue

        # replies are produced asynchronously; poll briefly for new outbound messages
        new: list[dict] = []
        for _ in range(240):
            time.sleep(0.25)
            new = [m for m in history() if m["id"] not in seen and m["direction"] == "out"]
            if new:
                time.sleep(0.3)  # allow multi-message replies to land
                new = [m for m in history() if m["id"] not in seen and m["direction"] == "out"]
                break
        for m in history():
            seen.add(m["id"])
        if not new:
            print("   (no reply yet - the agent may still be thinking)")
        for m in new:
            full = http.get(f"/api/messages/{m['id']}", headers={"X-API-Key": api_key}).json()
            opts, kind = render(full)
            if opts:
                options, reply_kind = opts, kind


if __name__ == "__main__":
    main()
