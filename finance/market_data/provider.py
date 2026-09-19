"""Provider contract for real-time market data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from finance.market_data.models import Bar, Quote, Trade


MarketEvent = Quote | Trade | Bar


class MarketDataProvider(Protocol):
    @property
    def name(self) -> str:
        ...

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def subscribe(self, symbols: Iterable[str]) -> None:
        ...

    def events(self) -> Iterable[MarketEvent]:
        ...
