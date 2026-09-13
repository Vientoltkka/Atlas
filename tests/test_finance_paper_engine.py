"""Tests de Atlas Finance V2.1: motor paper event-driven local."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from finance.paper.engine import Decision, PaperEngine
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
)
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore, StoreError


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 9, 13, 10, minute, 0, tzinfo=timezone.utc)


def _event(
    event_id: str,
    symbol: str,
    price: str,
    minute: int = 0,
    event_type: EventType = EventType.TICK,
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        symbol=symbol,
        price=Decimal(price),
        timestamp=_ts(minute),
        event_type=event_type,
    )


def _order(
    order_id: str,
    symbol: str,
    side: Side,
    qty: str,
    order_type: OrderType,
    limit: str | None = None,
    minute: int = 0,
) -> PaperOrder:
    return PaperOrder(
        symbol=symbol,
        side=side,
        qty=Decimal(qty),
        order_type=order_type,
        order_id=order_id,
        limit_price=None if limit is None else Decimal(limit),
        timestamp=_ts(minute),
    )


def _engine(cash: str = "10000", **kwargs) -> PaperEngine:
    return PaperEngine(PaperPortfolio(cash=Decimal(cash)), **kwargs)


def test_market_buy_fills_updates_cash_position_and_nav() -> None:
    engine = _engine()
    engine.process_event(_event("e1", "AAPL", "100"))
    decision = engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))

    assert decision.filled
    assert decision.fill is not None
    assert decision.fill.price == Decimal("100")
    assert engine.portfolio.cash == Decimal("9000")
    assert engine.portfolio.positions["AAPL"].qty == Decimal("10")
    assert engine.nav() == Decimal("10000")


def test_market_sell_reduces_position_and_increases_cash() -> None:
    engine = _engine()
    engine.process_event(_event("e1", "AAPL", "100"))
    engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    engine.process_event(_event("e2", "AAPL", "110", minute=1))
    decision = engine.execute_order(
        _order("o2", "AAPL", Side.SELL, "4", OrderType.MARKET, minute=1)
    )

    assert decision.filled
    assert decision.fill is not None
    assert decision.fill.price == Decimal("110")
    assert engine.portfolio.cash == Decimal("9000") + Decimal("440")
    assert engine.portfolio.positions["AAPL"].qty == Decimal("6")
    assert engine.nav() == Decimal("9440") + Decimal("660")


def test_buy_rejected_when_cash_insufficient_including_costs() -> None:
    engine = _engine(cash="900")
    engine.process_event(_event("e1", "AAPL", "100"))
    decision = engine.execute_order(
        _order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET)
    )
    assert decision.status is OrderStatus.REJECTED
    assert "efectivo insuficiente" in decision.reason
    assert engine.portfolio.cash == Decimal("900")
    assert engine.portfolio.positions == {}

    engine_with_costs = _engine(
        cash="1000", default_commission=Decimal("1"), default_slippage_bps=Decimal("10")
    )
    engine_with_costs.process_event(_event("e1", "AAPL", "100"))
    decision = engine_with_costs.execute_order(
        _order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET)
    )
    assert decision.status is OrderStatus.REJECTED
    assert "efectivo insuficiente" in decision.reason
    assert engine_with_costs.portfolio.cash == Decimal("1000")


def test_sell_rejected_when_position_insufficient() -> None:
    engine = _engine()
    engine.process_event(_event("e1", "AAPL", "100"))
    engine.execute_order(_order("o1", "AAPL", Side.BUY, "5", OrderType.MARKET))
    decision = engine.execute_order(
        _order("o2", "AAPL", Side.SELL, "10", OrderType.MARKET, minute=1)
    )
    assert decision.status is OrderStatus.REJECTED
    assert "posicion insuficiente" in decision.reason
    assert engine.portfolio.positions["AAPL"].qty == Decimal("5")

    empty = _engine()
    empty.process_event(_event("e1", "MSFT", "300"))
    decision = empty.execute_order(
        _order("o1", "MSFT", Side.SELL, "1", OrderType.MARKET)
    )
    assert decision.status is OrderStatus.REJECTED
    assert "posicion insuficiente" in decision.reason


def test_limit_order_pending_then_filled_on_later_tick() -> None:
    engine = _engine()
    engine.process_event(_event("e1", "AAPL", "100"))
    decision = engine.execute_order(
        _order("o1", "AAPL", Side.BUY, "10", OrderType.LIMIT, limit="90")
    )
    assert decision.status is OrderStatus.PENDING
    assert decision.fill is None
    assert engine.portfolio.cash == Decimal("10000")
    assert len(engine.pending_orders()) == 1

    assert engine.process_event(_event("e2", "AAPL", "95", minute=1)) is True
    assert len(engine.pending_orders()) == 1
    assert engine.portfolio.cash == Decimal("10000")

    assert engine.process_event(_event("e3", "AAPL", "89", minute=2)) is True
    assert engine.pending_orders() == ()
    assert engine.portfolio.cash == Decimal("10000") - Decimal("890")
    assert engine.portfolio.positions["AAPL"].qty == Decimal("10")
    assert engine.decision_for("o1") is not None
    assert engine.decision_for("o1").status is OrderStatus.FILLED


def test_limit_sell_fills_only_when_tick_reaches_limit() -> None:
    engine = _engine()
    engine.process_event(_event("e1", "AAPL", "100"))
    engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    decision = engine.execute_order(
        _order("o2", "AAPL", Side.SELL, "10", OrderType.LIMIT, limit="120", minute=1)
    )
    assert decision.status is OrderStatus.PENDING

    engine.process_event(_event("e2", "AAPL", "110", minute=2))
    assert len(engine.pending_orders()) == 1

    engine.process_event(_event("e3", "AAPL", "120", minute=3))
    assert engine.pending_orders() == ()
    assert engine.portfolio.cash == Decimal("9000") + Decimal("1200")
    assert engine.portfolio.positions == {}


def test_commission_and_slippage_are_explicit_in_fill() -> None:
    engine = _engine(
        default_commission=Decimal("1.50"), default_slippage_bps=Decimal("20")
    )
    engine.process_event(_event("e1", "AAPL", "100"))
    decision = engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))

    assert decision.filled
    fill = decision.fill
    assert isinstance(fill, PaperFill)
    assert fill.commission == Decimal("1.50")
    assert fill.slippage_bps == Decimal("20")
    assert fill.price == Decimal("100.2000")
    assert engine.portfolio.cash == Decimal("10000") - Decimal("1002") - Decimal("1.50")
    assert engine.portfolio.positions["AAPL"].avg_price == Decimal("100.2000")

    sell = _engine(
        cash="20000",
        default_commission=Decimal("1.50"),
        default_slippage_bps=Decimal("20"),
    )
    sell.process_event(_event("e1", "AAPL", "100"))
    sell.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    sell.process_event(_event("e2", "AAPL", "100", minute=1))
    decision = sell.execute_order(
        _order("o2", "AAPL", Side.SELL, "10", OrderType.MARKET, minute=1)
    )
    assert decision.fill.price == Decimal("99.8000")
    assert sell.portfolio.cash == Decimal("20000") - Decimal("1002") - Decimal("1.50") + Decimal("998") - Decimal("1.50")


def test_event_idempotency_does_not_duplicate_or_mutate() -> None:
    engine = _engine()
    assert engine.process_event(_event("e1", "AAPL", "100")) is True
    engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    state_before = engine.snapshot()
    ledger_len = len(state_before["ledger"])

    assert engine.process_event(_event("e1", "AAPL", "999")) is False
    state_after = engine.snapshot()
    assert state_after == state_before
    assert len(engine.ledger) == ledger_len
    assert engine.portfolio.cash == Decimal("9000")


def test_order_idempotency_does_not_duplicate_fills() -> None:
    engine = _engine()
    engine.process_event(_event("e1", "AAPL", "100"))
    first = engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    state_before = engine.snapshot()

    second = engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    assert second == first
    assert engine.snapshot() == state_before
    assert len(engine.fills) == 1
    assert engine.portfolio.cash == Decimal("9000")
    assert engine.portfolio.positions["AAPL"].qty == Decimal("10")


def test_ledger_records_event_decision_order_fill_with_reason_and_timestamp() -> None:
    engine = _engine(default_commission=Decimal("1"))
    engine.process_event(_event("e1", "AAPL", "100"))
    engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))

    kinds = [entry["kind"] for entry in engine.ledger]
    assert kinds == ["EVENT", "ORDER", "DECISION", "FILL"]
    for entry in engine.ledger:
        assert entry["reason"] != ""
        assert isinstance(entry["timestamp"], str)
        assert datetime.fromisoformat(str(entry["timestamp"])).tzinfo is not None
    fill_entry = engine.ledger[-1]
    assert fill_entry["payload"]["commission"] == "1.00"
    decision_entry = engine.ledger[-2]
    assert decision_entry["payload"]["status"] == "FILLED"


def test_store_roundtrip_preserves_portfolio_ledger_and_idempotency(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    engine = _engine(default_commission=Decimal("1"))
    engine.process_event(_event("e1", "AAPL", "100"))
    engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    engine.execute_order(_order("o2", "AAPL", Side.BUY, "10", OrderType.LIMIT, limit="90"))
    engine.process_event(_event("e2", "AAPL", "89", minute=1))
    store.save(engine.snapshot())

    restored = PaperEngine.from_snapshot(store.load())
    assert restored.snapshot() == engine.snapshot()
    assert restored.portfolio.cash == Decimal("8108")
    assert restored.portfolio.positions["AAPL"].qty == Decimal("20")
    assert restored.fills == engine.fills
    assert restored.ledger == engine.ledger

    assert restored.process_event(_event("e1", "AAPL", "100")) is False
    assert restored.snapshot() == engine.snapshot()
    assert restored.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET)) == (
        engine.decision_for("o1")
    )


def test_store_corrupt_state_raises_without_overwriting(tmp_path: Path) -> None:
    state_path = tmp_path / "finance_paper" / "state.json"
    state_path.parent.mkdir(parents=True)
    corrupted = "{ not valid json ]"
    state_path.write_text(corrupted, encoding="utf-8")

    store = PaperStore(state_path.parent)
    with pytest.raises(StoreError):
        store.load()
    assert state_path.read_text(encoding="utf-8") == corrupted

    with pytest.raises(StoreError):
        PaperFinanceService(store)
    assert state_path.read_text(encoding="utf-8") == corrupted


def test_store_invalid_version_raises_without_overwriting(tmp_path: Path) -> None:
    state_path = tmp_path / "finance_paper" / "state.json"
    state_path.parent.mkdir(parents=True)
    payload = json.dumps(
        {"schema_version": 999, "portfolio": {"cash": "1", "positions": []}}
    )
    state_path.write_text(payload, encoding="utf-8")

    store = PaperStore(state_path.parent)
    with pytest.raises(StoreError):
        store.load()
    with pytest.raises(StoreError):
        PaperFinanceService(store)
    assert state_path.read_text(encoding="utf-8") == payload


def test_store_rejects_frozen_models_and_safe_initialization(tmp_path: Path) -> None:
    event = _event("e1", "aapl ", "100")
    assert event.symbol == "AAPL"
    with pytest.raises(ValueError):
        MarketEvent(event_id="e2", symbol="AAPL", price=Decimal("-1"))
    with pytest.raises(ValueError):
        PaperOrder(
            symbol="AAPL",
            side=Side.BUY,
            qty=Decimal("0"),
            order_type=OrderType.MARKET,
            order_id="o1",
        )
    with pytest.raises(ValueError):
        PaperOrder(
            symbol="AAPL",
            side=Side.BUY,
            qty=Decimal("1"),
            order_type=OrderType.LIMIT,
            order_id="o2",
        )
    with pytest.raises(ValueError):
        PaperPosition(symbol="AAPL", qty=Decimal("1"), avg_price=Decimal("0"))

    store = PaperStore(tmp_path / "finance_paper")
    service = PaperFinanceService(store, starting_cash=Decimal("5000"))
    assert service.portfolio_summary()["cash"] == "5000.00"
    assert (tmp_path / "finance_paper" / "state.json").exists()
    with pytest.raises(StoreError):
        PaperStore(tmp_path / "missing_dir").load()


def test_service_facade_portfolio_events_and_confirmed_orders(tmp_path: Path) -> None:
    store = PaperStore(tmp_path / "finance_paper")
    service = PaperFinanceService(store)

    summary = service.portfolio_summary()
    assert summary["cash"] == "10000.00"
    assert summary["positions"] == {}

    service.record_market_event(_event("e1", "AAPL", "100"))
    decision = service.execute_confirmed_order(
        _order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET)
    )
    assert decision.filled

    summary = service.portfolio_summary()
    assert summary["cash"] == "9000.00"
    assert summary["nav"] == "10000.00"
    assert summary["positions"]["AAPL"]["qty"] == "10"
    assert service.pending_orders() == ()

    restored = PaperFinanceService(store)
    assert restored.portfolio_summary() == service.portfolio_summary()
    assert restored.engine.snapshot() == service.engine.snapshot()


def test_service_requires_explicit_confirmation_and_agent_stays_read_only() -> None:
    engine = _engine()
    decision = engine.execute_order(_order("o1", "AAPL", Side.BUY, "10", OrderType.MARKET))
    assert decision.status is OrderStatus.REJECTED
    assert "sin precio de referencia" in decision.reason

    source = Path("agents/finance_agent.py").read_text(encoding="utf-8").casefold()
    assert "finance.paper" not in source
    assert "no ejecutes compras" in source


def test_paper_stack_imports_only_stdlib_and_finance_modules() -> None:
    forbidden = {
        "requests",
        "urllib",
        "urllib2",
        "httpx",
        "aiohttp",
        "socket",
        "http",
        "websockets",
        "boto3",
        "ccxt",
        "alpaca",
        "yfinance",
        "websocket",
    }
    package_dir = Path(__file__).resolve().parent.parent / "finance"
    py_files = sorted(package_dir.rglob("*.py"))
    assert len(py_files) >= 5

    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = {(node.module or "").split(".")[0]} - {""}
            else:
                continue
            assert roots.isdisjoint(forbidden), f"{path.name} importa {roots}"
            assert roots <= {
                "__future__",
                "ast",
                "dataclasses",
                "datetime",
                "decimal",
                "enum",
                "json",
                "os",
                "pathlib",
                "tempfile",
                "typing",
                "uuid",
                "finance",
            }, f"{path.name} importa modulos inesperados: {roots}"
