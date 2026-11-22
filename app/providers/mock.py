"""Deterministic mock providers. The same query gives the same results (seeded RNG), and offer ids
encode the query, so an offer can be re-resolved later without a cache (survives restarts)."""

import hashlib
import random
from datetime import date as Date

from app.providers.base import FlightProvider, FoodProvider, TrainProvider

CITIES = {  # IATA code -> (city, main railway station code)
    "DEL": ("Delhi", "NDLS"), "BOM": ("Mumbai", "CSMT"), "BLR": ("Bengaluru", "SBC"),
    "MAA": ("Chennai", "MAS"), "CCU": ("Kolkata", "HWH"), "HYD": ("Hyderabad", "SC"),
    "GOI": ("Goa", "MAO"), "PNQ": ("Pune", "PUNE"), "JAI": ("Jaipur", "JP"),
    "LKO": ("Lucknow", "LKO"), "AMD": ("Ahmedabad", "ADI"), "COK": ("Kochi", "ERS"),
}  # fmt: skip
ALIASES = {"new delhi": "DEL", "bombay": "BOM", "bangalore": "BLR", "madras": "MAA", "calcutta": "CCU",
           "gurgaon": "DEL", "gurugram": "DEL", "noida": "DEL", "cochin": "COK"}  # fmt: skip
STATIONS = {stn: iata for iata, (_, stn) in CITIES.items()}
SUPPORTED = ", ".join(c for c, _ in CITIES.values())


def resolve_city(value: str) -> str | None:
    """Accept 'Delhi', 'del', 'NDLS', 'New Delhi' -> 'DEL'."""
    v = value.strip().lower()
    if v.upper() in CITIES:
        return v.upper()
    if v.upper() in STATIONS:
        return STATIONS[v.upper()]
    if v in ALIASES:
        return ALIASES[v]
    for code, (city, _) in CITIES.items():
        if city.lower() == v or (len(v) >= 3 and city.lower().startswith(v)):
            return code
    return None


def _route(origin: str, destination: str, date: str) -> tuple[str, str]:
    o, d = resolve_city(origin), resolve_city(destination)
    if not o or not d:
        raise ValueError(f"Unknown city. Supported cities: {SUPPORTED}")
    if o == d:
        raise ValueError("Origin and destination are the same")
    if Date.fromisoformat(date) < Date.today():
        raise ValueError("Date is in the past")
    return o, d


def _rng(*parts: object) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))


def _hhmm(minutes: int) -> str:
    """Clock time; adds a (+1) day marker when the time falls on a later day (overnight journeys)."""
    days = minutes // (24 * 60)
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}" + (f" (+{days})" if days else "")


class MockFlightProvider:
    AIRLINES = [("6E", "IndiGo"), ("AI", "Air India"), ("QP", "Akasa Air"), ("SG", "SpiceJet"), ("IX", "AI Express")]

    def search(self, origin: str, destination: str, date: str, adults: int = 1) -> list[dict]:
        o, d = _route(origin, destination, date)
        r = _rng("flight", o, d, date)
        out = []
        for i in range(6):
            code, name = r.choice(self.AIRLINES)
            dep = r.randrange(5 * 60, 23 * 60, 5)
            stops = 0 if r.random() < 0.7 else 1
            dur = r.randrange(70, 200, 5) + stops * 75
            fare = r.randrange(3200, 11000, 10) - stops * 600
            out.append({
                "offer_id": f"FL:{o}:{d}:{date}:{adults}:{i}",
                "airline": name, "flight_no": f"{code}-{r.randint(100, 999)}",
                "from": o, "to": d, "date": date, "depart": _hhmm(dep), "arrive": _hhmm(dep + dur),
                "duration_min": dur, "stops": stops, "fare_per_adult": fare, "adults": adults,
                "total": fare * adults, "baggage": "15kg check-in + 7kg cabin",
            })  # fmt: skip
        return sorted(out, key=lambda x: x["total"])

    def get_offer(self, offer_id: str) -> dict | None:
        try:
            _, o, d, date, adults, _ = offer_id.split(":")
            return next((x for x in self.search(o, d, date, int(adults)) if x["offer_id"] == offer_id), None)
        except ValueError:
            return None


class MockTrainProvider:
    NAMES = ["Rajdhani Express", "Shatabdi Express", "Duronto Express", "Vande Bharat Express",
             "Garib Rath", "Superfast Express", "Tejas Express"]  # fmt: skip
    CLASSES = {"1A": 4.2, "2A": 2.5, "3A": 1.8, "CC": 1.3, "SL": 0.7}

    def search(self, origin: str, destination: str, date: str, travel_class: str | None = None) -> list[dict]:
        o, d = _route(origin, destination, date)
        r = _rng("train", o, d, date)
        out = []
        for i in range(5):
            dep = r.randrange(0, 24 * 60, 5)
            dur = r.randrange(6 * 60, 26 * 60, 5)
            base = r.randrange(500, 900, 5)
            classes = {}
            for cls, mult in self.CLASSES.items():
                seats = r.choice([0, 2, 8]) if cls == "1A" else r.choice([0, 0, 3, 12, 48, 120])
                status = f"AVAILABLE-{seats}" if seats else f"WL {r.randint(4, 60)}"
                classes[cls] = {"fare": int(base * mult), "status": status, "offer_id": f"TR:{o}:{d}:{date}:{i}:{cls}"}
            if travel_class and travel_class.upper() in classes:
                classes = {travel_class.upper(): classes[travel_class.upper()]}
            out.append({
                "train_no": str(r.randint(12001, 22999)), "name": r.choice(self.NAMES),
                "from": CITIES[o][1], "to": CITIES[d][1], "date": date,
                "depart": _hhmm(dep), "arrive": _hhmm(dep + dur), "duration_min": dur, "classes": classes,
            })  # fmt: skip
        return sorted(out, key=lambda x: x["depart"])

    def get_offer(self, offer_id: str) -> dict | None:
        try:
            _, o, d, date, _, cls = offer_id.split(":")
            trains = self.search(o, d, date)
        except ValueError:
            return None
        for t in trains:
            c = t["classes"].get(cls)
            if c and c["offer_id"] == offer_id:
                base = {k: v for k, v in t.items() if k != "classes"}
                return {**base, "class": cls, "fare": c["fare"], "status": c["status"], "offer_id": offer_id}
        return None


class MockFoodProvider:
    RESTAURANTS = [
        {"id": "R1", "name": "Biryani Blues", "cuisine": "Biryani, Mughlai", "veg": False, "eta_min": 35, "rating": 4.3,
         "menu": [("Chicken Dum Biryani", 329, False), ("Veg Biryani", 249, True), ("Mutton Biryani", 429, False),
                  ("Raita", 49, True), ("Gulab Jamun (2)", 89, True)]},
        {"id": "R2", "name": "Pizza Piazza", "cuisine": "Pizza, Italian", "veg": False, "eta_min": 30, "rating": 4.1,
         "menu": [("Margherita (M)", 299, True), ("Farmhouse (M)", 399, True), ("Chicken Tikka Pizza (M)", 449, False),
                  ("Garlic Bread", 149, True), ("Coke 500ml", 60, True)]},
        {"id": "R3", "name": "Saravana Tiffins", "cuisine": "South Indian", "veg": True, "eta_min": 25, "rating": 4.5,
         "menu": [("Masala Dosa", 129, True), ("Idli Vada Combo", 99, True), ("Mini Meals", 199, True),
                  ("Filter Coffee", 49, True)]},
        {"id": "R4", "name": "Punjabi Tadka", "cuisine": "North Indian, Thali", "veg": False, "eta_min": 40, "rating": 4.2,
         "menu": [("Paneer Butter Masala", 279, True), ("Butter Chicken", 349, False), ("Dal Makhani", 229, True),
                  ("Butter Naan", 45, True), ("Veg Thali", 249, True)]},
        {"id": "R5", "name": "Green Bowl", "cuisine": "Salads, Healthy", "veg": True, "eta_min": 20, "rating": 4.4,
         "menu": [("Quinoa Salad Bowl", 299, True), ("Paneer Wrap", 199, True), ("Cold Pressed Juice", 149, True)]},
    ]  # fmt: skip

    @staticmethod
    def _public(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "menu"}

    def search(self, query: str = "", veg_only: bool = False) -> list[dict]:
        words = [w for w in query.lower().split() if len(w) > 2]
        out = []
        for r in self.RESTAURANTS:
            if veg_only and not r["veg"]:
                continue
            haystack = " ".join([r["name"], r["cuisine"], *(m[0] for m in r["menu"])]).lower()
            if words and not any(w in haystack for w in words):
                continue
            out.append(self._public(r))
        return out or ([self._public(r) for r in self.RESTAURANTS if r["veg"] or not veg_only])

    def menu(self, restaurant_id: str) -> dict | None:
        r = next((x for x in self.RESTAURANTS if x["id"] == restaurant_id.upper()), None)
        if not r:
            return None
        items = [{"item_id": f"{r['id']}-{i}", "name": n, "price": p, "veg": v} for i, (n, p, v) in enumerate(r["menu"], 1)]
        return {**self._public(r), "items": items}


flights: FlightProvider = MockFlightProvider()
trains: TrainProvider = MockTrainProvider()
food: FoodProvider = MockFoodProvider()
