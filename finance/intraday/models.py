"""Normalized intraday signal models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class IntradaySignalAction(str, Enum):
    CANDIDATE = "CANDIDATE"
    NO_ACTION = "NO_ACTION"


@dataclass(frozen=True)
class IntradayObservation:
    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    timestamp: datetime
    volume: Decimal | None = None

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol is required")
        object.__setattr__(self, "symbol", symbol)

        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        if self.price <= 0 or self.bid <= 0 or self.ask <= 0:
            raise ValueError("prices must be positive")

        if self.ask < self.bid:
            raise ValueError("ask cannot be below bid")

        if self.volume is not None and self.volume < 0:
            raise ValueError("volume cannot be negative")


@dataclass(frozen=True)
class IntradayFeatures:
    symbol: str
    timestamp: datetime
    observations: int
    last_price: Decimal
    return_short: Decimal
    return_long: Decimal
    acceleration: Decimal
    realized_volatility: Decimal
    relative_spread: Decimal


@dataclass(frozen=True)
class IntradaySignal:
    symbol: str
    timestamp: datetime
    action: IntradaySignalAction
    reasons: tuple[str, ...]
    features: IntradayFeatures
