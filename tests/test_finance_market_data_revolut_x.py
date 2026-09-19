from datetime import datetime, timezone
from decimal import Decimal

from finance.market_data.revolut_x import (
    RevolutXPublicMarketDataProvider,
)


def test_revolut_x_normalizes_public_ticker(monkeypatch) -> None:
    provider = RevolutXPublicMarketDataProvider()

    payload = {
        "data": [
            {
                "symbol": "BTC/USD",
                "bid": "60000.10",
                "ask": "60001.20",
                "mid": "60000.65",
                "last_price": "60000.50",
            }
        ],
        "metadata": {
            "timestamp": 1789826400000,
        },
    }

    monkeypatch.setattr(
        provider,
        "_request_tickers",
        lambda: payload,
    )

    provider.connect()
    provider.subscribe(["BTC-USD"])

    events = list(provider.events())

    assert len(events) == 1

    quote = events[0]

    assert quote.symbol == "BTC-USD"
    assert quote.bid == Decimal("60000.10")
    assert quote.ask == Decimal("60001.20")
    assert quote.provider == "REVOLUT_X_PUBLIC"
    assert quote.timestamp == datetime.fromtimestamp(
        1789826400,
        tz=timezone.utc,
    )


def test_revolut_x_does_not_emit_unsubscribed_symbols(
    monkeypatch,
) -> None:
    provider = RevolutXPublicMarketDataProvider()

    payload = {
        "data": [
            {
                "symbol": "BTC/USD",
                "bid": "60000",
                "ask": "60001",
            },
            {
                "symbol": "ETH/USD",
                "bid": "3000",
                "ask": "3001",
            },
        ],
        "metadata": {
            "timestamp": 1789826400000,
        },
    }

    monkeypatch.setattr(
        provider,
        "_request_tickers",
        lambda: payload,
    )

    provider.connect()
    provider.subscribe(["BTC-USD"])

    events = list(provider.events())

    assert len(events) == 1
    assert events[0].symbol == "BTC-USD"


def test_revolut_x_requires_subscription() -> None:
    provider = RevolutXPublicMarketDataProvider()
    provider.connect()

    try:
        list(provider.events())
    except RuntimeError as exc:
        assert "no symbols subscribed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_revolut_x_has_no_execution_or_credentials_dependency() -> None:
    import finance.market_data.revolut_x as module

    source = open(
        module.__file__,
        encoding="utf-8",
    ).read()

    assert "finance.execution" not in source
    assert "ExecutionIntent" not in source
    assert "BrokerAdapter" not in source

    assert "X-Revx-API-Key" not in source
    assert "X-Revx-Signature" not in source
