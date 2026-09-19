"""Normalized provider-neutral market-data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class MarketDataKind(str, Enum):
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    BAR = "BAR"


class MarketDataQuality(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    INVALID = "INVALID"


def _require_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if not value:
        raise ValueError("symbol is required")
    return value


def _require_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


def _positive(value: Decimal, field: str) -> None:
    if value <= 0:
        raise ValueError(f"{field} must be positive")


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: datetime
    provider: str
    sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_symbol(self.symbol))
        object.__setattr__(self, "timestamp", _require_timestamp(self.timestamp))
        _positive(self.bid, "bid")
        _positive(self.ask, "ask")
        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")
        if not self.provider.strip():
            raise ValueError("provider is required")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence cannot be negative")


@dataclass(frozen=True)
class Trade:
    symbol: str
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    provider: str
    sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_symbol(self.symbol))
        object.__setattr__(self, "timestamp", _require_timestamp(self.timestamp))
        _positive(self.price, "price")
        _positive(self.quantity, "quantity")
        if not self.provider.strip():
            raise ValueError("provider is required")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence cannot be negative")


@dataclass(frozen=True)
class Bar:
    symbol: str
    interval: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    start: datetime
    end: datetime
    provider: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_symbol(self.symbol))
        object.__setattr__(self, "start", _require_timestamp(self.start))
        object.__setattr__(self, "end", _require_timestamp(self.end))

        if not self.interval.strip():
            raise ValueError("interval is required")
        if self.end <= self.start:
            raise ValueError("bar end must be after start")

        _positive(self.open, "open")
        _positive(self.high, "high")
        _positive(self.low, "low")
        _positive(self.close, "close")

        if self.volume < 0:
            raise ValueError("volume cannot be negative")

        if self.high < max(self.open, self.close, self.low):
            raise ValueError("invalid bar high")

        if self.low > min(self.open, self.close, self.high):
            raise ValueError("invalid bar low")

        if not self.provider.strip():
            raise ValueError("provider is required")
