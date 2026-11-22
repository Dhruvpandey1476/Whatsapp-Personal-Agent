"""Live information tools: web search, reading a web page, and weather.

No API keys needed: DuckDuckGo's HTML endpoint (with Wikipedia as a fallback) for search, and
Open-Meteo for weather. Everything here is real, live data, unlike the mock booking providers.
"""

import html
import ipaddress
import logging
import re
import socket
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import settings

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
TIMEOUT = httpx.Timeout(12.0)


def _text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _ddg_url(href: str) -> str:
    """DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<real url>."""
    if "uddg=" in href:
        return unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
    return href


async def _duckduckgo(query: str, limit: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=UA, follow_redirects=True) as http:
        r = await http.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "in-en"})
    r.raise_for_status()
    if r.status_code != 200:  # 202 = bot challenge page
        return []
    links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
    out = []
    for i, (href, title) in enumerate(links[:limit]):
        url = _ddg_url(html.unescape(href))
        if "duckduckgo.com/y.js" in url:  # ads
            continue
        out.append({"title": _text(title), "url": url, "snippet": _text(snippets[i]) if i < len(snippets) else ""})
    return out


async def _wikipedia(query: str, limit: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=UA) as http:
        r = await http.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit})  # fmt: skip
    r.raise_for_status()
    return [{"title": x["title"], "url": "https://en.wikipedia.org/wiki/" + x["title"].replace(" ", "_"),
             "snippet": _text(x.get("snippet", ""))} for x in r.json().get("query", {}).get("search", [])]  # fmt: skip


async def _tavily(query: str, limit: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as http:
        r = await http.post("https://api.tavily.com/search", json={
            "api_key": settings.tavily_api_key, "query": query, "max_results": limit, "include_answer": True})  # fmt: skip
    r.raise_for_status()
    data = r.json()
    results = [{"title": x["title"], "url": x["url"], "snippet": x.get("content", "")[:400]} for x in data.get("results", [])]
    if data.get("answer"):
        results.insert(0, {"title": "Summary", "url": "", "snippet": data["answer"]})
    return results


async def _google_news(query: str, limit: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as http:
        r = await http.get("https://news.google.com/rss/search",
                           params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})  # fmt: skip
    r.raise_for_status()
    out = []
    for item in re.findall(r"<item>(.*?)</item>", r.text, re.S)[:limit]:
        get = lambda tag: _text(m.group(1)) if (m := re.search(rf"<{tag}>(.*?)</{tag}>", item, re.S)) else ""  # noqa: E731
        out.append({"title": get("title"), "url": get("link"), "snippet": f"Published {get('pubDate')}"})
    return out


async def web_search(query: str, limit: int = 5) -> dict:
    """Try each backend in order of quality; the first one that returns results wins."""
    backends = [("duckduckgo", _duckduckgo), ("google_news", _google_news), ("wikipedia", _wikipedia)]
    if settings.tavily_api_key:
        backends.insert(0, ("tavily", _tavily))
    for name, fn in backends:
        try:
            results = await fn(query, limit)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("%s search failed: %s", name, exc)
            continue
        if results:
            return {"source": name, "results": results}
    raise ValueError("Web search is unavailable right now")


def _is_public(url: str) -> bool:
    """Block localhost / private network targets so the agent can't be used to probe internal services."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


async def read_webpage(url: str, max_chars: int = 4000) -> dict:
    if not _is_public(url):
        raise ValueError("Only public http(s) web pages can be read")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=UA, follow_redirects=True) as http:
        r = await http.get(url)
    if not _is_public(str(r.url)):
        raise ValueError("Page redirected to a non-public address")
    r.raise_for_status()
    body = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", r.text)
    title = _text(m.group(1)) if (m := re.search(r"(?is)<title[^>]*>(.*?)</title>", body)) else ""
    text = re.sub(r"\s+", " ", _text(body))
    return {"url": str(r.url), "title": title, "text": text[:max_chars], "truncated": len(text) > max_chars}


WEATHER_CODES = {0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
                 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
                 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
                 81: "rain showers", 82: "violent showers", 95: "thunderstorm", 96: "thunderstorm with hail",
                 99: "thunderstorm with hail"}  # fmt: skip


# Regions people ask about that geocoders don't treat as cities ("Goa" -> Genoa, Italy).
REGIONS = {
    "goa": {"name": "Goa (Panaji)", "country": "India", "latitude": 15.4909, "longitude": 73.8278},
    "north goa": {"name": "North Goa (Calangute)", "country": "India", "latitude": 15.5439, "longitude": 73.7553},
    "south goa": {"name": "South Goa (Margao)", "country": "India", "latitude": 15.2832, "longitude": 73.9862},
    "kerala": {"name": "Kerala (Kochi)", "country": "India", "latitude": 9.9312, "longitude": 76.2673},
    "ladakh": {"name": "Ladakh (Leh)", "country": "India", "latitude": 34.1526, "longitude": 77.5771},
    "kashmir": {"name": "Kashmir (Srinagar)", "country": "India", "latitude": 34.0837, "longitude": 74.7973},
    "manali": {"name": "Manali", "country": "India", "latitude": 32.2432, "longitude": 77.1892},
    "andaman": {"name": "Andaman (Port Blair)", "country": "India", "latitude": 11.6234, "longitude": 92.7265},
}  # fmt: skip


def pick_place(places: list[dict], country: str = "") -> dict:
    """Honour an explicit country; otherwise keep the geocoder's population ranking, but prefer a
    *sizeable* Indian match (so 'Hyderabad' is India, while 'Dubai' stays UAE, not a village in UP)."""
    if country:
        c = country.lower()
        match = [p for p in places if c in (p.get("country", "").lower(), p.get("country_code", "").lower())]
        if match:
            return match[0]
    india = [p for p in places if p.get("country_code") == "IN" and (p.get("population") or 0) >= 50000]
    return india[0] if india else places[0]


async def weather(city: str, days: int = 3) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as http:
        name, _, country = (x.strip() for x in city.partition(","))
        p = REGIONS.get(name.lower())
        if p is None:
            g = await http.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": name, "count": 10})
            g.raise_for_status()
            places = g.json().get("results") or []
            if not places:
                raise ValueError(f"Couldn't find a place called {city}")
            p = pick_place(places, country)
        f = await http.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": max(1, min(days, 7)),
        })  # fmt: skip
        f.raise_for_status()
    d, c = f.json()["daily"], f.json()["current"]
    return {
        "place": ", ".join(x for x in (p.get("name"), p.get("admin1"), p.get("country")) if x),
        "now": {"temp_c": c["temperature_2m"], "feels_like_c": c["apparent_temperature"],
                "humidity": c["relative_humidity_2m"], "wind_kmh": c["wind_speed_10m"],
                "sky": WEATHER_CODES.get(c["weather_code"], "unknown")},
        "daily": [{"date": d["time"][i], "min_c": d["temperature_2m_min"][i], "max_c": d["temperature_2m_max"][i],
                   "rain_chance": d["precipitation_probability_max"][i],
                   "sky": WEATHER_CODES.get(d["weather_code"][i], "unknown")} for i in range(len(d["time"]))],
        "source": "open-meteo.com (live)",
    }  # fmt: skip
