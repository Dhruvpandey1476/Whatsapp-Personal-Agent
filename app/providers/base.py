"""Provider seam. The agent only talks to these interfaces, so swapping a mock for a real API
(Amadeus/Duffel for flights, an IRCTC-authorised agent API for trains, ONDC for food) means writing
one class. Tools, agent loop and the booking/payment flow stay untouched."""

from typing import Protocol


class FlightProvider(Protocol):
    def search(self, origin: str, destination: str, date: str, adults: int = 1) -> list[dict]: ...
    def get_offer(self, offer_id: str) -> dict | None: ...


class TrainProvider(Protocol):
    def search(self, origin: str, destination: str, date: str, travel_class: str | None = None) -> list[dict]: ...
    def get_offer(self, offer_id: str) -> dict | None: ...


class FoodProvider(Protocol):
    def search(self, query: str = "", veg_only: bool = False) -> list[dict]: ...
    def menu(self, restaurant_id: str) -> dict | None: ...
