"""Modelos inmutables del motor paper de Atlas Finance."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Mapping


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class EventType(Enum):
    TICK = "TICK"
    DECLARED = "DECLARED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_symbol(symbol: str) -> str:
    normalized = (symbol or "").strip().upper()
    if not normalized:
        raise ValueError("symbol vacio")
    if any(ch.isspace() for ch in normalized):
        raise ValueError(f"symbol con espacios: {symbol!r}")
    return normalized


def _decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} invalido: {value!r}")
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        result = Decimal(str(value))
    elif isinstance(value, str):
        try:
            result = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} invalido: {value!r}") from exc
    else:
        raise ValueError(f"{field_name} invalido: {value!r}")
    if not result.is_finite():
        raise ValueError(f"{field_name} no finito")
    return result


def _ensure_utc(timestamp: datetime) -> datetime:
    if not isinstance(timestamp, datetime):
        raise ValueError("timestamp debe ser datetime")
    if timestamp.tzinfo is None:
        raise ValueError("timestamp sin zona horaria")
    return timestamp.astimezone(timezone.utc)


def _ensure_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} vacio")
    return value.strip()


@dataclass(frozen=True)
class MarketEvent:
    event_id: str = field(default_factory=lambda: f"evt-{uuid.uuid4().hex}")
    symbol: str = ""
    price: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=utc_now)
    event_type: EventType = EventType.TICK
    source: str = "user_declared"
    provider_date: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _ensure_id(self.event_id, "event_id"))
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        price = _decimal(self.price, "price")
        if price <= 0:
            raise ValueError("price debe ser positivo")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "timestamp", _ensure_utc(self.timestamp))
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type invalido")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source vacio")
        object.__setattr__(self, "source", self.source.strip())
        if self.provider_date is not None:
            if isinstance(self.provider_date, datetime) or not isinstance(
                self.provider_date, date
            ):
                raise ValueError("provider_date invalida")


@dataclass(frozen=True)
class PaperOrder:
    symbol: str
    side: Side
    qty: Decimal
    order_type: OrderType
    order_id: str = field(default_factory=lambda: f"ord-{uuid.uuid4().hex}")
    limit_price: Decimal | None = None
    timestamp: datetime = field(default_factory=utc_now)
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _ensure_id(self.order_id, "order_id"))
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.side, Side):
            raise ValueError("side invalido")
        qty = _decimal(self.qty, "qty")
        if qty <= 0:
            raise ValueError("qty debe ser positivo")
        object.__setattr__(self, "qty", qty)
        if not isinstance(self.order_type, OrderType):
            raise ValueError("order_type invalido")
        if self.limit_price is not None:
            limit = _decimal(self.limit_price, "limit_price")
            if limit <= 0:
                raise ValueError("limit_price debe ser positivo")
            object.__setattr__(self, "limit_price", limit)
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("orden MARKET no admite limit_price")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("orden LIMIT exige limit_price")
        object.__setattr__(self, "timestamp", _ensure_utc(self.timestamp))
        if not isinstance(self.status, OrderStatus):
            raise ValueError("status invalido")


@dataclass(frozen=True)
class PaperFill:
    fill_id: str
    order_id: str
    event_id: str
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    commission: Decimal = Decimal("0")
    slippage_bps: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_id", _ensure_id(self.fill_id, "fill_id"))
        object.__setattr__(self, "order_id", _ensure_id(self.order_id, "order_id"))
        object.__setattr__(self, "event_id", _ensure_id(self.event_id, "event_id"))
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.side, Side):
            raise ValueError("side invalido")
        qty = _decimal(self.qty, "qty")
        if qty <= 0:
            raise ValueError("qty debe ser positivo")
        object.__setattr__(self, "qty", qty)
        price = _decimal(self.price, "price")
        if price <= 0:
            raise ValueError("price debe ser positivo")
        object.__setattr__(self, "price", price)
        commission = _decimal(self.commission, "commission")
        if commission < 0:
            raise ValueError("commission no puede ser negativa")
        object.__setattr__(self, "commission", commission)
        slippage = _decimal(self.slippage_bps, "slippage_bps")
        if slippage < 0:
            raise ValueError("slippage_bps no puede ser negativo")
        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "timestamp", _ensure_utc(self.timestamp))


@dataclass(frozen=True)
class PaperPosition:
    symbol: str
    qty: Decimal
    avg_price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        qty = _decimal(self.qty, "qty")
        if qty <= 0:
            raise ValueError("qty debe ser positivo")
        object.__setattr__(self, "qty", qty)
        avg = _decimal(self.avg_price, "avg_price")
        if avg <= 0:
            raise ValueError("avg_price debe ser positivo")
        object.__setattr__(self, "avg_price", avg)


@dataclass(frozen=True)
class PaperPortfolio:
    cash: Decimal = Decimal("10000")
    positions: Mapping[str, PaperPosition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cash = _decimal(self.cash, "cash")
        if cash < 0:
            raise ValueError("cash no puede ser negativo")
        object.__setattr__(self, "cash", cash)
        normalized: dict[str, PaperPosition] = {}
        for symbol, position in self.positions.items():
            if not isinstance(position, PaperPosition):
                raise ValueError("posicion invalida")
            normalized[normalize_symbol(symbol)] = position
        object.__setattr__(self, "positions", dict(normalized))

    def market_value(self, prices: Mapping[str, Decimal]) -> Decimal:
        total = Decimal("0")
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.avg_price)
            total += position.qty * price
        return total

    def nav(self, prices: Mapping[str, Decimal]) -> Decimal:
        return self.cash + self.market_value(prices)
