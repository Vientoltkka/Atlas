from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.intraday.research_runner import (
    LiveIntradayResearchRunner,
)
from finance.market_data.models import Quote


NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


class FakeProvider:
    def __init__(self, quotes):
        self.quotes = list(quotes)
        self.connected = False
        self.subscribed = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def subscribe(self, symbols):
        self.subscribed = tuple(symbols)

    def events(self):
        if not self.quotes:
            return iter(())

        return iter((self.quotes.pop(0),))


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def make_quote(second, bid="100", ask="100.10"):
    return Quote(
        symbol="BTC-USD",
        bid=Decimal(bid),
        ask=Decimal(ask),
        timestamp=NOW + timedelta(seconds=second),
        provider="TEST",
    )


def test_live_runner_collects_fresh_quotes(tmp_path):
    clock = FakeClock()

    provider = FakeProvider(
        [
            make_quote(0),
            make_quote(1),
            make_quote(2),
        ]
    )

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=3,
        poll_seconds=1,
        ledger_path=tmp_path / "research.jsonl",
        provider=provider,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        now_fn=lambda: NOW + timedelta(seconds=2),
    )

    stats = runner.run()

    assert stats.polls == 3
    assert stats.quotes_received == 3
    assert stats.quotes_accepted == 3
    assert stats.provider_errors == 0
    assert provider.subscribed == ("BTC-USD",)
    assert not provider.connected


def test_live_runner_rejects_stale_quote(tmp_path):
    clock = FakeClock()

    provider = FakeProvider(
        [
            Quote(
                symbol="BTC-USD",
                bid=Decimal("100"),
                ask=Decimal("101"),
                timestamp=NOW - timedelta(seconds=20),
                provider="TEST",
            )
        ]
    )

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=1,
        poll_seconds=1,
        ledger_path=tmp_path / "research.jsonl",
        provider=provider,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        now_fn=lambda: NOW,
    )

    stats = runner.run()

    assert stats.quotes_received == 1
    assert stats.quotes_accepted == 0
    assert stats.quotes_rejected_quality == 1


def test_live_runner_has_no_execution_dependency():
    import finance.intraday.research_runner as module

    source = open(
        module.__file__,
        encoding="utf-8-sig",
    ).read()

    assert "finance.execution" not in source
    assert "ExecutionIntent" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source
