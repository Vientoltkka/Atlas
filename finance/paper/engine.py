"""Motor paper determinista y auditable: eventos de mercado + ordenes paper.

Sin LLM, sin red y sin broker: cada decision se registra en un ledger con
razon y timestamp. Idempotencia estricta por event_id y order_id.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from finance.paper.models import (
    EventType,
    MarketEvent,
    OrderStatus,
    OrderType,
    PaperFill,
    PaperOrder,
    PaperPortfolio,
    PaperPosition,
    Side,
    utc_now,
)


@dataclass(frozen=True)
class Decision:
    order_id: str
    status: OrderStatus
    reason: str
    event_id: str | None = None
    fill: PaperFill | None = None

    @property
    def filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    @property
    def pending(self) -> bool:
        return self.status is OrderStatus.PENDING


LEDGER_EVENT = "EVENT"
LEDGER_DECISION = "DECISION"
LEDGER_ORDER = "ORDER"
LEDGER_FILL = "FILL"
LEDGER_CAPITAL = "CAPITAL"

STATE_VERSION = 1

SUPPORTED_VERSIONS = (1, 2)


def _q_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _q_price(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"))


class PaperEngine:
    def __init__(
        self,
        portfolio: PaperPortfolio | None = None,
        *,
        default_commission: Decimal = Decimal("0"),
        default_slippage_bps: Decimal = Decimal("0"),
    ) -> None:
        self._portfolio = portfolio or PaperPortfolio()
        self._default_commission = Decimal(default_commission)
        self._default_slippage_bps = Decimal(default_slippage_bps)
        self._seen_events: set[str] = set()
        self._orders: dict[str, PaperOrder] = {}
        self._decisions: dict[str, Decision] = {}
        self._pending: list[str] = []
        self._fills: list[PaperFill] = []
        self._last_prices: dict[str, Decimal] = {}
        self._ledger: list[dict[str, object]] = []
        self._seq = 0

    @property
    def portfolio(self) -> PaperPortfolio:
        return self._portfolio

    @property
    def fills(self) -> tuple[PaperFill, ...]:
        return tuple(self._fills)

    @property
    def ledger(self) -> tuple[dict[str, object], ...]:
        return tuple(self._ledger)

    def pending_orders(self) -> tuple[PaperOrder, ...]:
        return tuple(self._orders[order_id] for order_id in self._pending)

    def decision_for(self, order_id: str) -> Decision | None:
        return self._decisions.get(order_id)

    def last_prices(self) -> dict[str, Decimal]:
        return dict(self._last_prices)

    def defaults(self) -> dict[str, Decimal]:
        """Read-only execution defaults used to estimate order costs."""
        return {
            "commission": self._default_commission,
            "slippage_bps": self._default_slippage_bps,
        }

    def nav(self) -> Decimal:
        return _q_money(self._portfolio.nav(self._last_prices))

    def _append(
        self,
        kind: str,
        timestamp: datetime,
        payload: Mapping[str, object],
        reason: str = "",
    ) -> None:
        self._seq += 1
        self._ledger.append(
            {
                "seq": self._seq,
                "kind": kind,
                "timestamp": timestamp.isoformat(),
                "payload": dict(payload),
                "reason": reason,
            }
        )

    def transfer_cash(
        self, delta: Decimal, reason: str, timestamp: datetime | None = None
    ) -> PaperPortfolio:
        """Ajusta el efectivo paper por una transferencia explicita.

        Solo la invoca PaperFinanceService para asignaciones de capital
        paper entre modos. La operacion queda registrada en el ledger con
        kind CAPITAL. delta puede ser negativo, pero el efectivo resultante
        nunca puede ser negativo y las posiciones no se tocan.
        """
        amount = Decimal(delta)
        new_cash = _q_money(self._portfolio.cash + amount)
        if new_cash < 0:
            raise ValueError(
                f"la transferencia dejaria el efectivo paper negativo: "
                f"{self._portfolio.cash} {amount:+}"
            )
        self._portfolio = PaperPortfolio(
            cash=new_cash, positions=self._portfolio.positions
        )
        self._append(
            LEDGER_CAPITAL,
            timestamp or utc_now(),
            {"delta": str(amount), "cash_after": str(new_cash)},
            reason,
        )
        return self._portfolio

    def process_event(self, event: MarketEvent) -> bool:
        """Ingesta un MarketEvent. Devuelve False si ya fue procesado."""
        if event.event_id in self._seen_events:
            return False
        self._seen_events.add(event.event_id)
        self._last_prices[event.symbol] = event.price
        self._append(
            LEDGER_EVENT,
            event.timestamp,
            {
                "event_id": event.event_id,
                "symbol": event.symbol,
                "price": str(event.price),
                "event_type": event.event_type.value,
                "source": event.source,
                "provider_date": (
                    event.provider_date.isoformat()
                    if event.provider_date is not None
                    else None
                ),
            },
            "evento de mercado",
        )
        self._retry_pending_for(event)
        return True

    def execute_order(self, order: PaperOrder, event: MarketEvent | None = None) -> Decision:
        """Ejecuta una orden paper confirmada. Idempotente por order_id."""
        previous = self._decisions.get(order.order_id)
        if previous is not None:
            return previous

        self._orders[order.order_id] = order
        self._append(
            LEDGER_ORDER,
            order.timestamp,
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": str(order.qty),
                "order_type": order.order_type.value,
                "limit_price": None if order.limit_price is None else str(order.limit_price),
                "status": order.status.value,
            },
            "orden registrada",
        )
        decision = self._decide(order, event)
        self._decisions[order.order_id] = decision
        self._record(decision, event.timestamp if event is not None else order.timestamp)
        if decision.pending:
            self._pending.append(order.order_id)
        return decision

    def _decide(self, order: PaperOrder, event: MarketEvent | None) -> Decision:
        if order.status is not OrderStatus.PENDING:
            return Decision(order.order_id, order.status, "orden ya no esta pendiente")

        if order.order_type is OrderType.LIMIT:
            if event is None or event.symbol != order.symbol:
                return Decision(
                    order.order_id, OrderStatus.PENDING, "LIMIT pendiente sin tick aplicable"
                )
            if not self._limit_satisfied(order, event.price):
                return Decision(
                    order.order_id,
                    OrderStatus.PENDING,
                    f"tick {event.price} no satisface el limite {order.limit_price}",
                    event_id=event.event_id,
                )
            return self._fill(order, event, event.price)

        price = self._market_price(order.symbol, event)
        if price is None:
            return Decision(order.order_id, OrderStatus.REJECTED, "sin precio de referencia")
        if event is not None and event.symbol != order.symbol:
            return Decision(
                order.order_id, OrderStatus.REJECTED, "el evento no corresponde al simbolo"
            )
        return self._fill(order, event, price)

    def _market_price(self, symbol: str, event: MarketEvent | None) -> Decimal | None:
        if event is not None and event.symbol == symbol:
            return event.price
        return self._last_prices.get(symbol)

    def _limit_satisfied(self, order: PaperOrder, tick: Decimal) -> bool:
        assert order.limit_price is not None
        if order.side is Side.BUY:
            return tick <= order.limit_price
        return tick >= order.limit_price

    def _fill(
        self,
        order: PaperOrder,
        event: MarketEvent | None,
        reference_price: Decimal,
    ) -> Decision:
        slippage_bps = self._default_slippage_bps
        if order.side is Side.BUY:
            fill_price = reference_price * (Decimal("1") + slippage_bps / Decimal("10000"))
        else:
            fill_price = reference_price * (Decimal("1") - slippage_bps / Decimal("10000"))
        fill_price = _q_price(fill_price)
        commission = _q_money(self._default_commission)

        if order.side is Side.BUY:
            cost = fill_price * order.qty + commission
            if cost > self._portfolio.cash:
                return Decision(
                    order.order_id,
                    OrderStatus.REJECTED,
                    f"efectivo insuficiente: coste {cost} > cash {self._portfolio.cash}",
                    event_id=event.event_id if event else None,
                )
        else:
            position = self._portfolio.positions.get(order.symbol)
            held = position.qty if position is not None else Decimal("0")
            if held < order.qty:
                return Decision(
                    order.order_id,
                    OrderStatus.REJECTED,
                    f"posicion insuficiente: {held} < {order.qty}",
                    event_id=event.event_id if event else None,
                )

        fill = PaperFill(
            fill_id=f"fill-{uuid.uuid4().hex}",
            order_id=order.order_id,
            event_id=event.event_id if event is not None else "synthetic",
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            commission=commission,
            slippage_bps=slippage_bps,
            timestamp=event.timestamp if event is not None else order.timestamp,
        )
        self._apply_fill(fill)
        self._fills.append(fill)
        return Decision(
            order.order_id,
            OrderStatus.FILLED,
            "fill simulado",
            event_id=fill.event_id,
            fill=fill,
        )

    def _apply_fill(self, fill: PaperFill) -> None:
        cash = self._portfolio.cash
        positions = dict(self._portfolio.positions)
        position = positions.get(fill.symbol)
        if fill.side is Side.BUY:
            cash = cash - fill.price * fill.qty - fill.commission
            if position is None:
                positions[fill.symbol] = PaperPosition(fill.symbol, fill.qty, fill.price)
            else:
                total_qty = position.qty + fill.qty
                avg = (position.qty * position.avg_price + fill.qty * fill.price) / total_qty
                positions[fill.symbol] = PaperPosition(
                    fill.symbol, total_qty, _q_price(avg)
                )
        else:
            cash = cash + fill.price * fill.qty - fill.commission
            remaining = (position.qty if position else Decimal("0")) - fill.qty
            if remaining > 0 and position is not None:
                positions[fill.symbol] = PaperPosition(
                    fill.symbol, remaining, position.avg_price
                )
            else:
                positions.pop(fill.symbol, None)
        self._portfolio = PaperPortfolio(cash=_q_money(cash), positions=positions)
        self._orders[fill.order_id] = PaperOrder(
            symbol=self._orders[fill.order_id].symbol,
            side=self._orders[fill.order_id].side,
            qty=self._orders[fill.order_id].qty,
            order_type=self._orders[fill.order_id].order_type,
            order_id=fill.order_id,
            limit_price=self._orders[fill.order_id].limit_price,
            timestamp=self._orders[fill.order_id].timestamp,
            status=OrderStatus.FILLED,
        )

    def _retry_pending_for(self, event: MarketEvent) -> None:
        still_pending: list[str] = []
        for order_id in self._pending:
            order = self._orders[order_id]
            if order.symbol != event.symbol:
                still_pending.append(order_id)
                continue
            decision = self._decide(order, event)
            self._decisions[order.order_id] = decision
            self._record(decision, event.timestamp)
            if decision.pending:
                still_pending.append(order_id)
        self._pending = still_pending

    def _record(self, decision: Decision, timestamp: datetime) -> None:
        self._append(
            LEDGER_DECISION,
            timestamp,
            {
                "order_id": decision.order_id,
                "status": decision.status.value,
                "event_id": decision.event_id,
            },
            decision.reason,
        )
        fill = decision.fill
        if fill is not None:
            self._append(
                LEDGER_FILL,
                fill.timestamp,
                {
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "event_id": fill.event_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "qty": str(fill.qty),
                    "price": str(fill.price),
                    "commission": str(fill.commission),
                    "slippage_bps": str(fill.slippage_bps),
                },
                "fill simulado",
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": STATE_VERSION,
            "portfolio": {
                "cash": str(self._portfolio.cash),
                "positions": [
                    {
                        "symbol": position.symbol,
                        "qty": str(position.qty),
                        "avg_price": str(position.avg_price),
                    }
                    for position in self._portfolio.positions.values()
                ],
            },
            "seen_events": sorted(self._seen_events),
            "orders": [
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "qty": str(order.qty),
                    "order_type": order.order_type.value,
                    "limit_price": (
                        None if order.limit_price is None else str(order.limit_price)
                    ),
                    "timestamp": order.timestamp.isoformat(),
                    "status": order.status.value,
                }
                for order in self._orders.values()
            ],
            "pending": list(self._pending),
            "decisions": [
                {
                    "order_id": decision.order_id,
                    "status": decision.status.value,
                    "reason": decision.reason,
                    "event_id": decision.event_id,
                    "fill_id": None if decision.fill is None else decision.fill.fill_id,
                }
                for decision in self._decisions.values()
            ],
            "fills": [
                {
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "event_id": fill.event_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "qty": str(fill.qty),
                    "price": str(fill.price),
                    "commission": str(fill.commission),
                    "slippage_bps": str(fill.slippage_bps),
                    "timestamp": fill.timestamp.isoformat(),
                }
                for fill in self._fills
            ],
            "ledger": [dict(entry) for entry in self._ledger],
            "last_prices": {symbol: str(price) for symbol, price in self._last_prices.items()},
            "seq": self._seq,
            "defaults": {
                "commission": str(self._default_commission),
                "slippage_bps": str(self._default_slippage_bps),
            },
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, object]) -> "PaperEngine":
        version = data.get("schema_version")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"version de estado no soportada: {version!r}")
        portfolio_data = data["portfolio"]
        assert isinstance(portfolio_data, Mapping)
        positions = {
            position["symbol"]: PaperPosition(
                symbol=position["symbol"],
                qty=Decimal(position["qty"]),
                avg_price=Decimal(position["avg_price"]),
            )
            for position in portfolio_data["positions"]
        }
        portfolio = PaperPortfolio(cash=Decimal(portfolio_data["cash"]), positions=positions)
        defaults = data.get("defaults", {})
        assert isinstance(defaults, Mapping)
        engine = cls(
            portfolio,
            default_commission=Decimal(defaults.get("commission", "0")),
            default_slippage_bps=Decimal(defaults.get("slippage_bps", "0")),
        )
        engine._seen_events = set(data.get("seen_events", []))
        engine._last_prices = {
            symbol: Decimal(price)
            for symbol, price in dict(data.get("last_prices", {})).items()
        }
        for order_data in data.get("orders", []):
            assert isinstance(order_data, Mapping)
            limit = order_data.get("limit_price")
            order = PaperOrder(
                symbol=str(order_data["symbol"]),
                side=Side(str(order_data["side"])),
                qty=Decimal(str(order_data["qty"])),
                order_type=OrderType(str(order_data["order_type"])),
                order_id=str(order_data["order_id"]),
                limit_price=None if limit is None else Decimal(str(limit)),
                timestamp=datetime.fromisoformat(str(order_data["timestamp"])),
                status=OrderStatus(str(order_data["status"])),
            )
            engine._orders[order.order_id] = order
        engine._pending = [str(order_id) for order_id in data.get("pending", [])]
        for fill_data in data.get("fills", []):
            assert isinstance(fill_data, Mapping)
            engine._fills.append(
                PaperFill(
                    fill_id=str(fill_data["fill_id"]),
                    order_id=str(fill_data["order_id"]),
                    event_id=str(fill_data["event_id"]),
                    symbol=str(fill_data["symbol"]),
                    side=Side(str(fill_data["side"])),
                    qty=Decimal(str(fill_data["qty"])),
                    price=Decimal(str(fill_data["price"])),
                    commission=Decimal(str(fill_data.get("commission", "0"))),
                    slippage_bps=Decimal(str(fill_data.get("slippage_bps", "0"))),
                    timestamp=datetime.fromisoformat(str(fill_data["timestamp"])),
                )
            )
        engine._ledger = [dict(entry) for entry in data.get("ledger", [])]
        fills_by_id = {fill.fill_id: fill for fill in engine._fills}
        for decision_data in data.get("decisions", []):
            assert isinstance(decision_data, Mapping)
            fill_id = decision_data.get("fill_id")
            engine._decisions[str(decision_data["order_id"])] = Decision(
                order_id=str(decision_data["order_id"]),
                status=OrderStatus(str(decision_data["status"])),
                reason=str(decision_data.get("reason", "")),
                event_id=(
                    None if decision_data.get("event_id") is None
                    else str(decision_data["event_id"])
                ),
                fill=None if fill_id is None else fills_by_id.get(str(fill_id)),
            )
        engine._seq = int(data.get("seq", 0))
        return engine
