from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.intraday.collector import IntradayResearchCollector
from finance.intraday.evaluation import IntradaySignalEvaluator
from finance.intraday.models import IntradaySignalAction
from finance.intraday.persistence import IntradayResearchLedger
from finance.intraday.service import IntradaySignalService
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


def candidate_quotes() -> list[Quote]:
    return [
        quote(0, "100.00"),
        quote(1, "100.02"),
        quote(2, "100.04"),
        quote(3, "100.08"),
        quote(4, "100.20"),
        quote(5, "100.45"),
    ]


def test_collector_waits_for_minimum_observations(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
    )

    items = candidate_quotes()

    for item in items[:-1]:
        result = collector.ingest_quote(item)
        assert result.accepted_observation
        assert result.signal is None
        assert not result.recorded_signal

    records = ledger.records()
    assert all(
        record["type"] == "OBSERVATION"
        for record in records
    )


def test_collector_records_candidate_signal(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
    )

    result = None

    for item in candidate_quotes():
        result = collector.ingest_quote(item)

    assert result is not None
    assert result.signal is not None
    assert result.signal.action is IntradaySignalAction.CANDIDATE
    assert result.recorded_signal

    records = ledger.records()

    signals = [
        record
        for record in records
        if record["type"] == "SIGNAL"
    ]

    observations = [
        record
        for record in records
        if record["type"] == "OBSERVATION"
    ]

    assert len(observations) == 6
    assert len(signals) == 1
    assert signals[0]["symbol"] == "BTC-USD"
    assert signals[0]["action"] == "CANDIDATE"
    assert signals[0]["strategy_version"] == "intraday-momentum-v1"


def test_collector_does_not_record_no_action_by_default(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
    )

    flat = [
        quote(0, "100"),
        quote(1, "100"),
        quote(2, "100"),
        quote(3, "100"),
        quote(4, "100"),
        quote(5, "100"),
    ]

    result = None

    for item in flat:
        result = collector.ingest_quote(item)

    assert result is not None
    assert result.signal is not None
    assert result.signal.action is IntradaySignalAction.NO_ACTION
    assert not result.recorded_signal
    records = ledger.records()
    assert all(
        record["type"] == "OBSERVATION"
        for record in records
    )


def test_collector_can_record_no_action_for_research(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
        record_no_action=True,
    )

    flat = [
        quote(0, "100"),
        quote(1, "100"),
        quote(2, "100"),
        quote(3, "100"),
        quote(4, "100"),
        quote(5, "100"),
    ]

    result = None

    for item in flat:
        result = collector.ingest_quote(item)

    assert result is not None
    assert result.signal is not None
    assert result.signal.action is IntradaySignalAction.NO_ACTION
    assert result.recorded_signal

    records = ledger.records()

    signals = [
        record
        for record in records
        if record["type"] == "SIGNAL"
    ]

    observations = [
        record
        for record in records
        if record["type"] == "OBSERVATION"
    ]

    assert len(observations) == 6
    assert len(signals) == 1
    assert signals[0]["action"] == "NO_ACTION"


def test_collector_has_no_execution_dependency() -> None:
    import finance.intraday.collector as collector

    source = open(
        collector.__file__,
        encoding="utf-8-sig",
    ).read()

    assert "finance.execution" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source
    assert "ExecutionIntent(" not in source


def test_collector_persists_observations(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
    )

    collector.ingest_quote(quote(0, "100"))

    records = ledger.records()

    assert len(records) == 1
    assert records[0]["type"] == "OBSERVATION"
    assert records[0]["symbol"] == "BTC-USD"
    assert Decimal(records[0]["price"]) == Decimal("100")


def test_collector_records_completed_outcome_once(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
        evaluator=IntradaySignalEvaluator(
            horizons_minutes=(1,),
            maximum_lateness_seconds=30,
        ),
    )

    for item in candidate_quotes():
        collector.ingest_quote(item)

    result = collector.ingest_quote(
        quote(6, "100.55")
    )

    assert result.recorded_outcomes == 1

    records = ledger.records()

    outcomes = [
        item
        for item in records
        if item["type"] == "OUTCOME"
    ]

    assert len(outcomes) == 1
    assert outcomes[0]["horizon_minutes"] == 1
    assert outcomes[0]["entry_price"] == "100.45"
    assert outcomes[0]["exit_price"] == "100.55"

    collector.ingest_quote(
        quote(7, "100.60")
    )

    records = ledger.records()

    outcomes = [
        item
        for item in records
        if item["type"] == "OUTCOME"
    ]

    assert len(outcomes) == 1


def test_collector_rejects_duplicate_observation(tmp_path) -> None:
    ledger = IntradayResearchLedger(tmp_path / "intraday.jsonl")

    collector = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger,
    )

    item = quote(0, "100")

    first = collector.ingest_quote(item)
    second = collector.ingest_quote(item)

    assert first.accepted_observation
    assert not second.accepted_observation

    observations = [
        record
        for record in ledger.records()
        if record["type"] == "OBSERVATION"
    ]

    assert len(observations) == 1


def test_collector_recovers_candidate_and_outcomes_after_restart(
    tmp_path,
) -> None:
    path = tmp_path / "intraday.jsonl"

    ledger1 = IntradayResearchLedger(path)

    collector1 = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger1,
        evaluator=IntradaySignalEvaluator(
            horizons_minutes=(1,),
            maximum_lateness_seconds=30,
        ),
    )

    for item in candidate_quotes():
        collector1.ingest_quote(item)

    collector1.ingest_quote(
        quote(6, "100.55")
    )

    outcomes_before = [
        record
        for record in ledger1.records()
        if record["type"] == "OUTCOME"
    ]

    assert len(outcomes_before) == 1

    # Simulate a full Atlas/process restart.
    ledger2 = IntradayResearchLedger(path)

    collector2 = IntradayResearchCollector(
        signal_service=IntradaySignalService(),
        ledger=ledger2,
        evaluator=IntradaySignalEvaluator(
            horizons_minutes=(1,),
            maximum_lateness_seconds=30,
        ),
    )

    collector2.ingest_quote(
        quote(7, "100.60")
    )

    records = ledger2.records()

    outcomes_after = [
        record
        for record in records
        if record["type"] == "OUTCOME"
    ]

    assert len(outcomes_after) == 1

    observations = [
        record
        for record in records
        if record["type"] == "OBSERVATION"
    ]

    assert len(observations) == 8
