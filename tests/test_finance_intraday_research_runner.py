from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.intraday.research_runner import (
    DEFAULT_LEDGER,
    DEFAULT_V2_LEDGER,
    LiveIntradayResearchRunner,
    V1_STRATEGY_VERSION,
    _parser,
)
from finance.intraday.time_replay import TIME_BASED_STRATEGY_VERSION
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


class SequenceProvider:
    def __init__(self, actions):
        self.actions = list(actions)
        self.events_calls = 0
        self.connected = False

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def subscribe(self, symbols):
        self.subscribed = tuple(symbols)

    def events(self):
        self.events_calls += 1
        action = self.actions.pop(0) if self.actions else None
        if isinstance(action, BaseException):
            raise action
        return iter(()) if action is None else iter((action,))


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


def test_live_runner_recovers_timeout_without_duplicate_processing(
    tmp_path,
    capsys,
):
    clock = FakeClock()
    quote = make_quote(0)
    provider = SequenceProvider(
        [TimeoutError("read operation timed out"), quote, quote]
    )

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=2,
        poll_seconds=1,
        ledger_path=tmp_path / "research.jsonl",
        provider=provider,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        now_fn=lambda: NOW + timedelta(seconds=2),
        provider_retry_backoff=(0,),
    )

    stats = runner.run()

    assert stats.provider_retry_attempts == 1
    assert stats.provider_retries_recovered == 1
    assert stats.provider_errors_finales == 0
    assert stats.provider_errors == 0
    assert stats.quotes_received == 2
    assert stats.quotes_accepted == 1
    assert stats.quotes_rejected_duplicate == 1
    assert provider.events_calls == 3
    assert "[provider-retry-recovered]" in capsys.readouterr().out


def test_live_runner_continues_after_retry_exhaustion(tmp_path):
    clock = FakeClock()
    provider = SequenceProvider(
        [
            TimeoutError("timeout 1"),
            TimeoutError("timeout 2"),
            TimeoutError("timeout 3"),
            TimeoutError("timeout 4"),
            make_quote(1),
        ]
    )

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=2,
        poll_seconds=1,
        ledger_path=tmp_path / "research.jsonl",
        provider=provider,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        now_fn=lambda: NOW + timedelta(seconds=2),
        provider_retry_backoff=(0,),
    )

    stats = runner.run()

    assert stats.provider_retry_attempts == 3
    assert stats.provider_retries_recovered == 0
    assert stats.provider_errors_finales == 1
    assert stats.provider_errors == 1
    assert stats.quotes_accepted == 1
    assert stats.polls == 2


def test_live_runner_does_not_retry_after_capture_window(tmp_path):
    clock = FakeClock()
    provider = SequenceProvider([TimeoutError("timeout"), make_quote(1)])

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=0.5,
        poll_seconds=1,
        ledger_path=tmp_path / "research.jsonl",
        provider=provider,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        now_fn=lambda: NOW,
        provider_retry_backoff=(1,),
    )

    stats = runner.run()

    assert provider.events_calls == 1
    assert stats.provider_retry_attempts == 0
    assert stats.provider_errors_finales == 1


@pytest.mark.parametrize("error", [ValueError("invalid quote"), RuntimeError("bug")])
def test_live_runner_does_not_retry_non_transient_provider_errors(
    tmp_path,
    error,
):
    clock = FakeClock()
    provider = SequenceProvider([error])

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=1,
        poll_seconds=1,
        ledger_path=tmp_path / "research.jsonl",
        provider=provider,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        provider_retry_backoff=(0,),
    )

    with pytest.raises(type(error), match=str(error)):
        runner.run()

    assert provider.events_calls == 1
    assert clock.value == 0


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


def test_parser_keeps_v1_defaults():
    args = _parser().parse_args([])

    assert args.strategy_version == V1_STRATEGY_VERSION
    assert args.symbol == "BTC-USD"
    assert args.minutes == 60.0
    assert args.poll_seconds == 5.0
    assert args.ledger is None
    assert DEFAULT_LEDGER.name == "live_research.jsonl"


def test_v2_parser_and_runner_use_time_based_collector(tmp_path):
    args = _parser().parse_args(
        ["--strategy-version", TIME_BASED_STRATEGY_VERSION]
    )
    ledger_path = tmp_path / "v2.jsonl"

    runner = LiveIntradayResearchRunner(
        symbol=args.symbol,
        duration_seconds=args.minutes * 60,
        poll_seconds=args.poll_seconds,
        ledger_path=ledger_path,
        strategy_version=args.strategy_version,
        provider=FakeProvider([]),
    )

    collector = runner._collector
    ledger = collector._ledger
    assert ledger.strategy_version == TIME_BASED_STRATEGY_VERSION
    assert collector._record_no_action is True
    assert any(
        record["type"] == "CONFIGURATION"
        for record in ledger.records()
    )


def test_v2_default_ledger_path():
    assert DEFAULT_V2_LEDGER == (
        DEFAULT_V2_LEDGER.parent / "time_based_v2_research.jsonl"
    )


def test_runner_rejects_mixed_ledger_strategy(tmp_path):
    ledger_path = tmp_path / "mixed.jsonl"
    ledger_path.write_text(
        '{"strategy_version":"intraday-momentum-v1",'
        '"type":"OBSERVATION"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible strategy_version"):
        LiveIntradayResearchRunner(
            symbol="BTC-USD",
            duration_seconds=600,
            poll_seconds=5,
            ledger_path=ledger_path,
            strategy_version=TIME_BASED_STRATEGY_VERSION,
            provider=FakeProvider([]),
        )


def test_v2_without_ledger_uses_v2_default(monkeypatch, tmp_path):
    import finance.intraday.research_runner as module

    default_path = tmp_path / "time_based_v2_research.jsonl"
    monkeypatch.setattr(module, "DEFAULT_V2_LEDGER", default_path)

    runner = LiveIntradayResearchRunner(
        symbol="BTC-USD",
        duration_seconds=600,
        poll_seconds=5,
        strategy_version=TIME_BASED_STRATEGY_VERSION,
        provider=FakeProvider([]),
    )

    assert runner._collector._ledger.path == default_path
