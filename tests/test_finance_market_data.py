from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.market_data.models import Bar, MarketDataQuality, Quote, Trade
from finance.market_data.quality import MarketDataQualityGate
from finance.market_data.store import MarketDataStore


NOW = datetime(2026, 9, 19, 14, 0, tzinfo=timezone.utc)


def quote(*, seconds_old: int = 0, sequence: int | None = 1) -> Quote:
    return Quote(
        symbol="VUSA.AMS",
        bid=Decimal("125.90"),
        ask=Decimal("126.00"),
        timestamp=NOW - timedelta(seconds=seconds_old),
        provider="TEST",
        sequence=sequence,
    )


def trade(*, seconds_old: int = 0, sequence: int | None = 1) -> Trade:
    return Trade(
        symbol="VUSA.AMS",
        price=Decimal("125.95"),
        quantity=Decimal("2"),
        timestamp=NOW - timedelta(seconds=seconds_old),
        provider="TEST",
        sequence=sequence,
    )


def test_quote_requires_timezone_and_valid_spread() -> None:
    with pytest.raises(ValueError):
        Quote(
            symbol="TEST",
            bid=Decimal("101"),
            ask=Decimal("100"),
            timestamp=NOW,
            provider="TEST",
        )

    with pytest.raises(ValueError):
        Quote(
            symbol="TEST",
            bid=Decimal("100"),
            ask=Decimal("101"),
            timestamp=NOW.replace(tzinfo=None),
            provider="TEST",
        )


def test_quality_gate_accepts_fresh_quote() -> None:
    decision = MarketDataQualityGate().evaluate(
        quote(seconds_old=2),
        now=NOW,
    )

    assert decision.accepted
    assert decision.quality is MarketDataQuality.FRESH


def test_quality_gate_rejects_stale_quote() -> None:
    decision = MarketDataQualityGate().evaluate(
        quote(seconds_old=20),
        now=NOW,
    )

    assert not decision.accepted
    assert decision.quality is MarketDataQuality.STALE
    assert decision.reason == "STALE_DATA"


def test_quality_gate_rejects_future_timestamp() -> None:
    future = Quote(
        symbol="TEST",
        bid=Decimal("10"),
        ask=Decimal("11"),
        timestamp=NOW + timedelta(seconds=10),
        provider="TEST",
    )

    decision = MarketDataQualityGate().evaluate(future, now=NOW)

    assert not decision.accepted
    assert decision.quality is MarketDataQuality.INVALID
    assert decision.reason == "FUTURE_TIMESTAMP"


def test_store_rejects_duplicate_or_lower_sequence() -> None:
    store = MarketDataStore()

    first = quote(seconds_old=2, sequence=10)
    same_sequence = quote(seconds_old=1, sequence=10)
    lower_sequence = quote(seconds_old=0, sequence=9)
    higher_sequence_same_time = Quote(
        symbol="VUSA.AMS",
        bid=Decimal("125.91"),
        ask=Decimal("126.01"),
        timestamp=first.timestamp,
        provider="TEST",
        sequence=11,
    )

    assert store.put_quote(first)
    assert not store.put_quote(same_sequence)
    assert not store.put_quote(lower_sequence)
    assert store.put_quote(higher_sequence_same_time)
    assert store.quote("vusa.ams") == higher_sequence_same_time


def test_store_without_sequence_uses_timestamp() -> None:
    store = MarketDataStore()

    old = quote(seconds_old=5, sequence=None)
    new = quote(seconds_old=1, sequence=None)

    assert store.put_quote(old)
    assert store.put_quote(new)
    assert store.quote("VUSA.AMS") == new


def test_store_keeps_latest_trade() -> None:
    store = MarketDataStore()

    old = trade(seconds_old=5, sequence=1)
    new = trade(seconds_old=1, sequence=2)

    assert store.put_trade(old)
    assert store.put_trade(new)
    assert store.trade("VUSA.AMS") == new


def test_bar_validates_ohlc_and_store_ordering() -> None:
    store = MarketDataStore()

    first = Bar(
        symbol="VUSA.AMS",
        interval="1m",
        open=Decimal("125"),
        high=Decimal("127"),
        low=Decimal("124"),
        close=Decimal("126"),
        volume=Decimal("1000"),
        start=NOW - timedelta(minutes=2),
        end=NOW - timedelta(minutes=1),
        provider="TEST",
    )

    second = Bar(
        symbol="VUSA.AMS",
        interval="1m",
        open=Decimal("126"),
        high=Decimal("128"),
        low=Decimal("125"),
        close=Decimal("127"),
        volume=Decimal("900"),
        start=NOW - timedelta(minutes=1),
        end=NOW,
        provider="TEST",
    )

    assert store.put_bar(first)
    assert store.put_bar(second)
    assert not store.put_bar(first)
    assert store.bar("VUSA.AMS", "1m") == second


def test_no_execution_dependency() -> None:
    import finance.market_data.models as models
    import finance.market_data.provider as provider
    import finance.market_data.quality as quality
    import finance.market_data.store as store

    for module in (models, provider, quality, store):
        source = open(module.__file__, encoding="utf-8").read()
        assert "finance.execution" not in source
        assert "PaperFinanceService" not in source
        assert "BrokerAdapter" not in source
