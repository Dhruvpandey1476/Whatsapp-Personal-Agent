"""One-command local demo with the browser agent (it needs more RAM than Render's free plan).

    python scripts/demo.py          start: tunnel + server (browser on) + Meta webhook -> this laptop
                                    Ctrl+C stops everything and points the webhook back to the cloud
    python scripts/demo.py stop     only point the webhook back to the cloud (if the window was closed)

Reads credentials from .env; never prints secrets. The cloud URL comes from DEMO_RESTORE_WEBHOOK_URL in .env.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
GRAPH = "https://graph.facebook.com/v23.0"
PORT = 8000


def load_env() -> dict:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env()


def need(key: str) -> str:
    if not ENV.get(key):
        sys.exit(f"Missing {key} in .env")
    return ENV[key]


def point_webhook(url: str, quiet: bool = False) -> None:
    """Point both Meta webhook settings (app default + WhatsApp-account override) at `url`. Meta verifies it."""
    token, secret = need("WHATSAPP_ACCESS_TOKEN"), need("WHATSAPP_APP_SECRET")
    waba, verify = need("WHATSAPP_BUSINESS_ACCOUNT_ID"), need("WHATSAPP_VERIFY_TOKEN")
    info = httpx.get(f"{GRAPH}/debug_token", params={"input_token": token, "access_token": token}, timeout=20).json()
    app_id = info.get("data", {}).get("app_id")
    if not app_id:
        sys.exit("Could not read the app id from the WhatsApp token (is the token valid?)")
    r1 = httpx.post(f"{GRAPH}/{app_id}/subscriptions", timeout=90, data={
        "access_token": f"{app_id}|{secret}", "object": "whatsapp_business_account",
        "callback_url": url, "verify_token": verify, "fields": "messages"})  # fmt: skip
    r2 = httpx.post(f"{GRAPH}/{waba}/subscribed_apps", timeout=90, headers={"Authorization": f"Bearer {token}"},
                    data={"override_callback_uri": url, "verify_token": verify})  # fmt: skip
    ok = r1.status_code == 200 and r2.status_code == 200
    if not quiet:
        print(f"{'✅' if ok else '❌'} Meta webhook -> {url}" + ("" if ok else f"\n   {r1.text[:200]}\n   {r2.text[:200]}"))
    if not ok:
        raise RuntimeError("webhook update failed")


def restore() -> None:
    cloud = need("DEMO_RESTORE_WEBHOOK_URL")
    base = cloud.rsplit("/webhook", 1)[0]
    print(f"Waking the cloud server ({base}) so Meta can verify it…")
    for _ in range(12):
        try:
            if httpx.get(f"{base}/health", timeout=20).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    point_webhook(cloud)


def find_cloudflared() -> str:
    for c in (shutil.which("cloudflared"), r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
              r"C:\Program Files\cloudflared\cloudflared.exe"):  # fmt: skip
        if c and Path(c).exists():
            return c
    sys.exit("cloudflared not found. Install it: winget install Cloudflare.cloudflared")


def start() -> None:
    procs: list[subprocess.Popen] = []
    switched = False
    try:
        print("1/3 Starting the public tunnel…")
        tunnel = subprocess.Popen([find_cloudflared(), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace")  # fmt: skip
        procs.append(tunnel)
        url = None
        deadline = time.time() + 60
        while time.time() < deadline and not url:
            line = tunnel.stdout.readline()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
            url = m.group(0) if m else None
        if not url:
            sys.exit("Tunnel did not start (no URL within 60s)")
        print(f"    {url}")

        print("2/3 Starting the server with the browser agent enabled…")
        env = {**os.environ, "BROWSER_ENABLED": "true", "PUBLIC_BASE_URL": url}
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)], cwd=ROOT, env=env)
        procs.append(server)
        for _ in range(60):  # wait for the server itself (localhost always resolves)
            try:
                if httpx.get(f"http://localhost:{PORT}/health", timeout=3).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            sys.exit("The server didn't start. See the errors above.")

        # Brand-new tunnel names can take a minute or two to resolve, and this PC may cache a failed lookup.
        # What matters is whether META can reach the tunnel, and Meta verifies it on every webhook update,
        # so retry the update instead of testing the tunnel from here.
        print("3/3 Pointing Meta's webhook at this laptop (waiting for the tunnel to go live)…")
        deadline = time.time() + 180
        while True:
            try:
                point_webhook(f"{url}/webhook", quiet=True)
                break
            except RuntimeError:
                if time.time() > deadline:
                    sys.exit("Meta couldn't reach the tunnel after 3 minutes. Run demo.py again (a new tunnel is created).")
                print("    tunnel not reachable yet, retrying in 10s…")
                time.sleep(10)
        print(f"✅ Meta webhook -> {url}/webhook")
        switched = True
        print("\n🎬 Demo is LIVE. Message your WhatsApp test number. SkyTrip site:", f"{url}/skytrip")
        print("   Press Ctrl+C to stop and hand WhatsApp back to the cloud server.\n")
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        for p in reversed(procs):
            p.terminate()
        if switched:
            try:
                restore()
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Could not restore the webhook automatically ({exc}). Run: python scripts/demo.py stop")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    restore() if sys.argv[1:] == ["stop"] else start()
