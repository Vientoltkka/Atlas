from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.intraday.collector import IntradayResearchCollector
from finance.intraday.evaluation import IntradaySignalEvaluator
from finance.intraday.models import (
    IntradayFeatures,
    IntradaySignal,
    IntradaySignalAction,
)
from finance.intraday.persistence import IntradayResearchLedger
from finance.intraday.service import TimeBasedIntradaySignalService
from finance.intraday.signal_engine import IntradaySignalEngine
from finance.intraday.time_features import TimeBasedIntradayFeatureEngine
from finance.intraday.time_replay import TIME_BASED_STRATEGY_VERSION, load_observations
from finance.intraday.outcome_analysis import load_research_event_report
from finance.market_data.models import Quote


START = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def quote(minute: int, price: str) -> Quote:
    value = Decimal(price)
    return Quote(
        symbol="BTC-USD",
        bid=value - Decimal("0.05"),
        ask=value + Decimal("0.05"),
        timestamp=START + timedelta(minutes=minute),
        provider="test",
    )


def rising_quotes() -> list[Quote]:
    return [
        quote(0, "100.00"),
        quote(1, "100.02"),
        quote(2, "100.04"),
        quote(3, "100.08"),
        quote(4, "100.20"),
        quote(5, "100.45"),
        quote(6, "100.80"),
    ]


def make_v2(path, *, evaluator=None):
    ledger = IntradayResearchLedger(
        path,
        strategy_version=TIME_BASED_STRATEGY_VERSION,
    )
    collector = IntradayResearchCollector.for_time_based_v2(
        ledger=ledger,
        evaluator=evaluator,
    )
    return collector, ledger


class CandidateService:
    configuration = {}

    def ingest(self, observation):
        return IntradaySignal(
            symbol=observation.symbol,
            timestamp=observation.timestamp,
            action=IntradaySignalAction.CANDIDATE,
            reasons=("TEST_POLICY",),
            features=IntradayFeatures(
                symbol=observation.symbol,
                timestamp=observation.timestamp,
                observations=1,
                last_price=observation.price,
                return_short=Decimal("0.01"),
                return_long=Decimal("0.01"),
                acceleration=Decimal("0.01"),
                realized_volatility=Decimal("0"),
                relative_spread=Decimal("0"),
            ),
            direction="LONG",
        )


class CandidateThenNoActionService(CandidateService):
    def __init__(self, candidates: int) -> None:
        self._candidates = candidates
        self._calls = 0

    def ingest(self, observation):
        self._calls += 1
        signal = super().ingest(observation)
        if self._calls <= self._candidates:
            return signal
        return IntradaySignal(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            action=IntradaySignalAction.NO_ACTION,
            reasons=("TEST_POLICY",),
            features=signal.features,
        )


def test_v2_capture_persists_observation_features_signal_and_config(tmp_path):
    collector, ledger = make_v2(tmp_path / "research.jsonl")

    results = [collector.ingest_quote(item) for item in rising_quotes()]
    records = ledger.records()
    signals = [item for item in records if item["type"] == "SIGNAL"]
    configurations = [
        item for item in records if item["type"] == "CONFIGURATION"
    ]

    assert all(item.accepted_observation for item in results)
    assert signals
    assert signals[-1]["strategy_version"] == TIME_BASED_STRATEGY_VERSION
    assert signals[-1]["direction"] == "LONG"
    assert len(configurations) == 1
    assert configurations[0]["capture_id"] == ledger.capture_id
    assert {
        key: configurations[0][key]
        for key in ("type", "strategy_version", "configuration")
    } == {
        "type": "CONFIGURATION",
        "strategy_version": TIME_BASED_STRATEGY_VERSION,
        "configuration": {
                "short_minutes": 1.0,
                "long_minutes": 5.0,
                "maximum_reference_lateness_seconds": 30,
                "volatility_step_seconds": 60,
                "minimum_short_return": "0.0010",
                "minimum_long_return": "0.0020",
                "minimum_acceleration": "0",
                "maximum_volatility": "0.0100",
                "maximum_relative_spread": "0.0030",
        },
    }


def test_v2_persists_no_action_but_not_insufficient_history(tmp_path):
    collector, ledger = make_v2(tmp_path / "research.jsonl")

    for minute in range(5):
        result = collector.ingest_quote(quote(minute, "100"))
        assert result.signal is None

    result = collector.ingest_quote(quote(5, "100"))
    assert result.signal is not None
    assert result.signal.action.value == "NO_ACTION"

    signals = [
        item for item in ledger.records() if item["type"] == "SIGNAL"
    ]
    assert len(signals) == 1
    assert signals[0]["action"] == "NO_ACTION"
    assert signals[0]["direction"] is None
    assert signals[0]["strategy_version"] == TIME_BASED_STRATEGY_VERSION


def test_v2_restart_does_not_duplicate_records_or_outcomes(tmp_path):
    path = tmp_path / "research.jsonl"
    evaluator = IntradaySignalEvaluator(
        horizons_minutes=(1,),
        maximum_lateness_seconds=30,
    )
    collector, ledger = make_v2(path, evaluator=evaluator)

    for item in rising_quotes()[:6]:
        collector.ingest_quote(item)
    collector.ingest_quote(rising_quotes()[6])
    before = ledger.records()

    restarted, ledger_after = make_v2(path, evaluator=evaluator)
    restarted.ingest_quote(rising_quotes()[6])

    assert ledger_after.records() == before
    assert sum(item["type"] == "SIGNAL" for item in before) == 2
    assert sum(item["type"] == "OUTCOME" for item in before) == 1


def test_v2_replay_from_jsonl_matches_capture(tmp_path):
    path = tmp_path / "research.jsonl"
    collector, ledger = make_v2(path)
    for item in rising_quotes():
        collector.ingest_quote(item)

    observations = load_observations(path)
    engine = TimeBasedIntradayFeatureEngine()
    signal_engine = IntradaySignalEngine()
    replayed = []
    for index in range(len(observations)):
        try:
            features = engine.calculate(observations[: index + 1])
        except ValueError:
            continue
        replayed.append(signal_engine.evaluate(features))

    captured = [
        item for item in ledger.records()
        if item["type"] == "SIGNAL"
    ]
    assert [item.action.value for item in replayed] == [
        item["action"] for item in captured
    ]
    assert [item.features.timestamp.isoformat() for item in replayed] == [
        item["timestamp"] for item in captured
    ]


def test_v2_modules_have_no_execution_dependency():
    import finance.intraday.collector as collector
    import finance.intraday.service as service

    for module in (collector, service):
        source = open(module.__file__, encoding="utf-8").read()
        assert "finance.execution" not in source
        assert "ExecutionIntent" not in source
        assert "BrokerAdapter" not in source
        assert "PaperFinanceService" not in source


def test_v2_candidates_within_cooldown_are_one_durable_event(tmp_path):
    path = tmp_path / "events.jsonl"
    ledger = IntradayResearchLedger(
        path,
        strategy_version=TIME_BASED_STRATEGY_VERSION,
    )
    collector = IntradayResearchCollector.for_time_based_v2(
        ledger=ledger,
        signal_service=CandidateService(),
    )

    for second in range(12):
        value = Decimal("100") + Decimal(second) / Decimal("100")
        collector.ingest_quote(
            Quote(
                symbol="BTC-USD",
                bid=value - Decimal("0.05"),
                ask=value + Decimal("0.05"),
                timestamp=START + timedelta(seconds=second),
                provider="test",
            )
        )

    records = ledger.records()
    events = [item for item in records if item["type"] == "EVENT"]
    updates = [item for item in records if item["type"] == "EVENT_UPDATE"]
    candidates = [
        item for item in records
        if item["type"] == "SIGNAL" and item["action"] == "CANDIDATE"
    ]
    assert len(events) == 1
    assert events[0]["direction"] == "LONG"
    assert updates[-1]["observation_count"] == 12
    assert candidates
    assert all(item["event_id"] == events[0]["id"] for item in candidates)
    assert len({item["event_id"] for item in candidates}) == 1


def test_v2_candidate_event_link_is_restored_without_reconstruction(tmp_path):
    path = tmp_path / "events.jsonl"
    collector, ledger = make_v2(path)
    for item in rising_quotes():
        collector.ingest_quote(item)

    candidate = next(
        item for item in ledger.records()
        if item["type"] == "SIGNAL" and item["action"] == "CANDIDATE"
    )
    assert candidate["event_id"] in {
        item["id"] for item in ledger.records() if item["type"] == "EVENT"
    }

    restarted, _ = make_v2(path)
    key = (candidate["symbol"], datetime.fromisoformat(candidate["timestamp"]))
    assert restarted._candidate_event_ids[key] == candidate["event_id"]


def test_v2_event_restores_open_cooldown_without_duplication(tmp_path):
    path = tmp_path / "events.jsonl"
    ledger = IntradayResearchLedger(
        path,
        strategy_version=TIME_BASED_STRATEGY_VERSION,
    )
    first = IntradayResearchCollector.for_time_based_v2(
        ledger=ledger,
        signal_service=CandidateService(),
    )
    first.ingest_quote(quote(0, "100"))

    restarted = IntradayResearchCollector.for_time_based_v2(
        ledger=IntradayResearchLedger(
            path,
            strategy_version=TIME_BASED_STRATEGY_VERSION,
        ),
        signal_service=CandidateService(),
    )
    restarted.ingest_quote(quote(1, "100.01"))

    events = [
        item for item in restarted._ledger.records()
        if item["type"] == "EVENT"
    ]
    assert len(events) == 1
    assert restarted._events[next(iter(restarted._events))].observation_count == 2


def test_v2_no_action_observations_complete_one_event_once(tmp_path):
    path = tmp_path / "future-events.jsonl"
    collector, ledger = make_v2(
        path,
        evaluator=IntradaySignalEvaluator(
            horizons_minutes=(1,),
            maximum_lateness_seconds=30,
        ),
    )
    # Use the real event horizons; the evaluator override above only limits
    # legacy OUTCOME records.
    collector._signal_service = CandidateThenNoActionService(candidates=4)

    for second in range(4):
        collector.ingest_quote(
            Quote(
                symbol="BTC-USD",
                bid=Decimal("99.95"),
                ask=Decimal("100.05"),
                timestamp=START + timedelta(seconds=second),
                provider="test",
            )
        )

    future = {
        1: "101",
        5: "99",
        15: "102",
        30: "98",
    }
    for minute, price in future.items():
        result = collector.ingest_quote(quote(minute, price))
        assert result.signal.action is IntradaySignalAction.NO_ACTION

    event = next(iter(collector._events.values()))
    assert event.observation_count == 4
    assert event.outcome_1m.future_price == Decimal("101")
    assert event.outcome_5m.future_price == Decimal("99")
    assert event.outcome_15m.future_price == Decimal("102")
    assert event.outcome_30m.future_price == Decimal("98")
    assert event.outcome_1m.gross_return == Decimal("0.01")
    assert event.outcome_5m.gross_return == Decimal("-0.01")
    assert event.outcome_15m.gross_return == Decimal("0.02")
    assert event.outcome_30m.gross_return == Decimal("-0.02")
    assert event.outcome_1m.net_return == Decimal("0.0090")
    assert event.outcome_5m.net_return == Decimal("-0.0110")

    updates = [
        item for item in ledger.records() if item["type"] == "EVENT_UPDATE"
    ]
    assert len(updates) == 7

    before = len(updates)
    collector.ingest_quote(quote(31, "98"))
    assert sum(
        item["type"] == "EVENT_UPDATE" for item in ledger.records()
    ) == before


def test_v2_restored_resolved_event_does_not_duplicate_update(tmp_path):
    path = tmp_path / "resolved-events.jsonl"
    service = CandidateThenNoActionService(candidates=1)
    ledger = IntradayResearchLedger(
        path,
        strategy_version=TIME_BASED_STRATEGY_VERSION,
    )
    collector = IntradayResearchCollector.for_time_based_v2(
        ledger=ledger,
        signal_service=service,
    )
    collector.ingest_quote(quote(0, "100"))
    collector.ingest_quote(quote(1, "101"))
    before = ledger.records()

    restarted = IntradayResearchCollector.for_time_based_v2(
        ledger=IntradayResearchLedger(
            path,
            strategy_version=TIME_BASED_STRATEGY_VERSION,
        ),
        signal_service=CandidateThenNoActionService(candidates=1),
    )

    assert ledger.records() == before
    assert restarted._events[next(iter(restarted._events))].outcome_1m.future_price == Decimal("101")


def test_v2_capture_ids_isolate_observations_and_reports(tmp_path):
    path = tmp_path / "isolated.jsonl"
    first_ledger = IntradayResearchLedger(
        path, strategy_version=TIME_BASED_STRATEGY_VERSION, capture_id="capture-a"
    )
    second_ledger = IntradayResearchLedger(
        path, strategy_version=TIME_BASED_STRATEGY_VERSION, capture_id="capture-b"
    )
    first = IntradayResearchCollector.for_time_based_v2(
        ledger=first_ledger, signal_service=CandidateService()
    )
    second = IntradayResearchCollector.for_time_based_v2(
        ledger=second_ledger, signal_service=CandidateService()
    )
    first.ingest_quote(quote(0, "100"))
    second.ingest_quote(quote(0, "200"))

    with pytest.raises(ValueError, match="capture-a.*capture-b"):
        load_observations(path)
    assert load_observations(path, capture_id="capture-a")[0].price == Decimal("100")
    assert load_observations(path, capture_id="capture-b")[0].price == Decimal("200")

    with pytest.raises(ValueError, match="capture_id must be selected"):
        load_research_event_report(
            path, strategy_version=TIME_BASED_STRATEGY_VERSION
        )
    assert load_research_event_report(
        path,
        strategy_version=TIME_BASED_STRATEGY_VERSION,
        capture_id="capture-a",
    ).capture_id == "capture-a"


def test_v2_capture_configuration_is_snapshot_per_capture(tmp_path):
    path = tmp_path / "configuration.jsonl"
    ledger = IntradayResearchLedger(
        path, strategy_version=TIME_BASED_STRATEGY_VERSION, capture_id="capture-a"
    )
    configuration = {"minimum_long_return": "0.0020", "nested": {"x": 1}}
    ledger.ensure_configuration(configuration)
    configuration["nested"]["x"] = 99
    stored = next(item for item in ledger.records() if item["type"] == "CONFIGURATION")
    assert stored["capture_id"] == "capture-a"
    assert stored["configuration"]["nested"]["x"] == 1


def test_legacy_ledger_remains_readable_without_inventing_capture_id(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        "{\"type\":\"OBSERVATION\",\"strategy_version\":\"v2\","
        "\"symbol\":\"BTC-USD\",\"price\":\"100\","
        "\"bid\":\"99.99\",\"ask\":\"100.01\","
        "\"timestamp\":\"2026-09-19T12:00:00+00:00\"}\n",
        encoding="utf-8",
    )

    observations = load_observations(path)

    assert len(observations) == 1
    assert observations[0].price == Decimal("100")


def test_capture_configuration_cannot_be_reused_with_different_snapshot(tmp_path):
    path = tmp_path / "configuration.jsonl"
    ledger = IntradayResearchLedger(
        path, strategy_version=TIME_BASED_STRATEGY_VERSION, capture_id="capture-a"
    )
    ledger.ensure_configuration({"threshold": "0.001"})

    with pytest.raises(ValueError, match="configuration snapshot differs"):
        ledger.ensure_configuration({"threshold": "0.002"})
