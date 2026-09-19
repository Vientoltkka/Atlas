"""Immutable contracts for the only new Finance execution path."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from finance.paper.models import OrderType, Side, _decimal, normalize_symbol


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionMode(Enum):
    PAPER = "PAPER"
    REAL = "REAL"


class ExecutionStatus(Enum):
    FILLED = "FILLED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExecutionIntent:
    symbol: str
    side: Side
    quantity: Decimal
    order_type: OrderType
    mode: ExecutionMode = ExecutionMode.PAPER
    reference_price: Decimal | None = None
    strategy: str = "manual"
    signal_id: str | None = None
    asset_class: str = "EQUITY"
    currency: str | None = None
    intent_id: str = field(default_factory=lambda: f"intent-{uuid.uuid4().hex}")
    created_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.side, Side) or not isinstance(self.order_type, OrderType):
            raise ValueError("side u order_type invalido")
        if not isinstance(self.mode, ExecutionMode):
            raise ValueError("mode invalido")
        quantity = _decimal(self.quantity, "quantity")
        if quantity <= 0:
            raise ValueError("quantity debe ser positiva")
        object.__setattr__(self, "quantity", quantity)
        if self.reference_price is not None:
            price = _decimal(self.reference_price, "reference_price")
            if price <= 0:
                raise ValueError("reference_price debe ser positivo")
            object.__setattr__(self, "reference_price", price)
        if not isinstance(self.strategy, str) or not self.strategy.strip():
            raise ValueError("strategy vacia")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at sin zona horaria")


@dataclass(frozen=True)
class RiskDecision:
    intent_id: str
    approved: bool
    reasons: tuple[str, ...]
    evaluated_at: datetime = field(default_factory=_now)
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    intent_id: str
    status: ExecutionStatus
    adapter: str
    mode: ExecutionMode
    timestamp: datetime
    symbol: str
    side: Side
    quantity: Decimal
    price: Decimal | None = None
    commission: Decimal | None = None
    currency: str | None = None
    reason: str = ""
