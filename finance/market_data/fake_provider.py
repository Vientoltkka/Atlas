"""Deterministic provider used to validate the market-data pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from finance.market_data.models import Bar, Quote, Trade


MarketEvent = Quote | Trade | Bar


class FakeMarketDataProvider:
    def __init__(
        self,
        events: Iterable[MarketEvent],
        *,
        fail_connect: bool = False,
        fail_events: bool = False,
    ) -> None:
        self._events = list(events)
        self._fail_connect = fail_connect
        self._fail_events = fail_events

        self.connected = False
        self.disconnected = False
        self.subscriptions: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return "FAKE"

    def connect(self) -> None:
        if self._fail_connect:
            raise ConnectionError("fake connect failure")
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True

    def subscribe(self, symbols: Iterable[str]) -> None:
        if not self.connected:
            raise RuntimeError("provider is not connected")

        self.subscriptions = tuple(
            dict.fromkeys(
                symbol.strip().upper()
                for symbol in symbols
                if symbol.strip()
            )
        )

    def events(self) -> Iterable[MarketEvent]:
        if not self.connected:
            raise RuntimeError("provider is not connected")

        if self._fail_events:
            raise ConnectionError("fake stream failure")

        yield from self._events
