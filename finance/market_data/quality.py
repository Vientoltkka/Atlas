"""Fail-closed freshness validation for market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from finance.market_data.models import (
    Bar,
    MarketDataQuality,
    Quote,
    Trade,
)


MarketEvent = Quote | Trade | Bar


@dataclass(frozen=True)
class QualityDecision:
    quality: MarketDataQuality
    reason: str
    age_seconds: float

    @property
    def accepted(self) -> bool:
        return self.quality is MarketDataQuality.FRESH


class MarketDataQualityGate:
    def __init__(
        self,
        *,
        max_quote_age: timedelta = timedelta(seconds=5),
        max_trade_age: timedelta = timedelta(seconds=10),
        max_bar_age: timedelta = timedelta(minutes=2),
        max_future_skew: timedelta = timedelta(seconds=2),
    ) -> None:
        self._max_quote_age = max_quote_age
        self._max_trade_age = max_trade_age
        self._max_bar_age = max_bar_age
        self._max_future_skew = max_future_skew

    def evaluate(
        self,
        event: MarketEvent,
        *,
        now: datetime,
    ) -> QualityDecision:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

        timestamp = event.end if isinstance(event, Bar) else event.timestamp
        age = now - timestamp

        if age < -self._max_future_skew:
            return QualityDecision(
                MarketDataQuality.INVALID,
                "FUTURE_TIMESTAMP",
                age.total_seconds(),
            )

        if isinstance(event, Quote):
            maximum = self._max_quote_age
        elif isinstance(event, Trade):
            maximum = self._max_trade_age
        else:
            maximum = self._max_bar_age

        if age > maximum:
            return QualityDecision(
                MarketDataQuality.STALE,
                "STALE_DATA",
                age.total_seconds(),
            )

        return QualityDecision(
            MarketDataQuality.FRESH,
            "OK",
            age.total_seconds(),
        )
