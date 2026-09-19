from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.market_data.fake_provider import FakeMarketDataProvider
from finance.market_data.models import Quote
from finance.market_data.quality import MarketDataQualityGate
from finance.market_data.runtime import MarketDataRuntime
from finance.market_data.store import MarketDataStore


NOW = datetime(2026, 9, 19, 15, 0, tzinfo=timezone.utc)


def make_quote(
    *,
    age_seconds: int = 0,
    sequence: int = 1,
) -> Quote:
    return Quote(
        symbol="VUSA.AMS",
        bid=Decimal("125.90"),
        ask=Decimal("126.00"),
        timestamp=NOW - timedelta(seconds=age_seconds),
        provider="FAKE",
        sequence=sequence,
    )


def runtime(provider: FakeMarketDataProvider) -> tuple[MarketDataRuntime, MarketDataStore]:
    store = MarketDataStore()

    service = MarketDataRuntime(
        provider=provider,
        quality_gate=MarketDataQualityGate(),
        store=store,
        clock=lambda: NOW,
    )

    return service, store


def test_runtime_connects_subscribes_ingests_and_disconnects() -> None:
    provider = FakeMarketDataProvider([make_quote()])
    service, store = runtime(provider)

    stats = service.run_once(["vusa.ams", "VUSA.AMS"])

    assert stats.received == 1
    assert stats.accepted == 1
    assert stats.provider_errors == 0

    assert provider.subscriptions == ("VUSA.AMS",)
    assert provider.disconnected
    assert store.quote("VUSA.AMS") is not None


def test_runtime_rejects_stale_market_data() -> None:
    provider = FakeMarketDataProvider(
        [make_quote(age_seconds=30)]
    )

    service, store = runtime(provider)
    stats = service.run_once(["VUSA.AMS"])

    assert stats.received == 1
    assert stats.accepted == 0
    assert stats.rejected_quality == 1
    assert store.quote("VUSA.AMS") is None


def test_runtime_rejects_duplicate_sequence() -> None:
    provider = FakeMarketDataProvider(
        [
            make_quote(sequence=10),
            make_quote(sequence=10),
        ]
    )

    service, store = runtime(provider)
    stats = service.run_once(["VUSA.AMS"])

    assert stats.received == 2
    assert stats.accepted == 1
    assert stats.rejected_ordering == 1
    assert store.quote("VUSA.AMS") is not None


def test_runtime_fails_closed_on_provider_connect_error() -> None:
    provider = FakeMarketDataProvider(
        [],
        fail_connect=True,
    )

    service, store = runtime(provider)
    stats = service.run_once(["VUSA.AMS"])

    assert stats.accepted == 0
    assert stats.provider_errors == 1
    assert store.quote("VUSA.AMS") is None


def test_runtime_fails_closed_on_stream_error_and_disconnects() -> None:
    provider = FakeMarketDataProvider(
        [],
        fail_events=True,
    )

    service, store = runtime(provider)
    stats = service.run_once(["VUSA.AMS"])

    assert stats.accepted == 0
    assert stats.provider_errors == 1
    assert provider.disconnected
    assert store.quote("VUSA.AMS") is None


def test_runtime_has_no_execution_dependency() -> None:
    import finance.market_data.runtime as market_runtime

    source = open(
        market_runtime.__file__,
        encoding="utf-8",
    ).read()

    assert "finance.execution" not in source
    assert "BrokerAdapter" not in source
    assert "ExecutionIntent" not in source
