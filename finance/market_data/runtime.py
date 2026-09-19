"""Provider-neutral ingestion runtime for Atlas market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from finance.market_data.models import Bar, Quote, Trade
from finance.market_data.provider import MarketDataProvider
from finance.market_data.quality import MarketDataQualityGate
from finance.market_data.store import MarketDataStore


MarketEvent = Quote | Trade | Bar


@dataclass(frozen=True)
class IngestionStats:
    received: int = 0
    accepted: int = 0
    rejected_quality: int = 0
    rejected_ordering: int = 0
    provider_errors: int = 0


class MarketDataRuntime:
    """Moves provider events through quality validation into market state.

    This layer has no execution dependency and cannot place orders.
    """

    def __init__(
        self,
        *,
        provider: MarketDataProvider,
        quality_gate: MarketDataQualityGate,
        store: MarketDataStore,
        clock: Callable[[], datetime],
    ) -> None:
        self._provider = provider
        self._quality_gate = quality_gate
        self._store = store
        self._clock = clock

    def ingest(self, event: MarketEvent) -> bool:
        decision = self._quality_gate.evaluate(
            event,
            now=self._clock(),
        )

        if not decision.accepted:
            return False

        if isinstance(event, Quote):
            return self._store.put_quote(event)

        if isinstance(event, Trade):
            return self._store.put_trade(event)

        if isinstance(event, Bar):
            return self._store.put_bar(event)

        return False

    def run_once(self, symbols: list[str]) -> IngestionStats:
        received = 0
        accepted = 0
        rejected_quality = 0
        rejected_ordering = 0
        provider_errors = 0

        connected = False

        try:
            self._provider.connect()
            connected = True
            self._provider.subscribe(symbols)

            for event in self._provider.events():
                received += 1

                decision = self._quality_gate.evaluate(
                    event,
                    now=self._clock(),
                )

                if not decision.accepted:
                    rejected_quality += 1
                    continue

                if isinstance(event, Quote):
                    stored = self._store.put_quote(event)
                elif isinstance(event, Trade):
                    stored = self._store.put_trade(event)
                elif isinstance(event, Bar):
                    stored = self._store.put_bar(event)
                else:
                    stored = False

                if stored:
                    accepted += 1
                else:
                    rejected_ordering += 1

        except Exception:
            provider_errors += 1

        finally:
            if connected:
                try:
                    self._provider.disconnect()
                except Exception:
                    provider_errors += 1

        return IngestionStats(
            received=received,
            accepted=accepted,
            rejected_quality=rejected_quality,
            rejected_ordering=rejected_ordering,
            provider_errors=provider_errors,
        )
