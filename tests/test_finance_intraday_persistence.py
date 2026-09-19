from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.intraday.evaluation import (
    IntradaySignalEvaluator,
)
from finance.intraday.features import IntradayFeatureEngine
from finance.intraday.models import (
    IntradayObservation,
    IntradaySignalAction,
)
from finance.intraday.persistence import IntradayResearchLedger
from finance.intraday.signal_engine import IntradaySignalEngine


START = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def obs(minute: int, price: str) -> IntradayObservation:
    value = Decimal(price)

    return IntradayObservation(
        symbol="BTC-USD",
        price=value,
        bid=value - Decimal("0.05"),
        ask=value + Decimal("0.05"),
        timestamp=START + timedelta(minutes=minute),
    )


def candidate():
    history = [
        obs(0, "100.00"),
        obs(1, "100.02"),
        obs(2, "100.04"),
        obs(3, "100.08"),
        obs(4, "100.20"),
        obs(5, "100.45"),
    ]

    features = IntradayFeatureEngine().calculate(history)
    signal = IntradaySignalEngine().evaluate(features)

    assert signal.action is IntradaySignalAction.CANDIDATE
    return signal


def test_ledger_persists_signal(tmp_path) -> None:
    ledger = IntradayResearchLedger(
        tmp_path / "intraday.jsonl"
    )

    signal = candidate()
    ledger.record_signal(signal)

    records = ledger.records()

    assert len(records) == 1
    assert records[0]["type"] == "SIGNAL"
    assert records[0]["symbol"] == "BTC-USD"
    assert records[0]["action"] == "CANDIDATE"
    assert records[0]["features"]["last_price"] == "100.45"


def test_ledger_persists_forward_outcome(tmp_path) -> None:
    ledger = IntradayResearchLedger(
        tmp_path / "intraday.jsonl"
    )

    signal = candidate()

    future = [
        obs(6, "100.55"),
        obs(7, "100.60"),
        obs(8, "100.70"),
        obs(9, "100.75"),
        obs(10, "100.80"),
    ]

    outcomes = IntradaySignalEvaluator(
        horizons_minutes=(5,)
    ).evaluate(signal, future)

    assert len(outcomes) == 1

    ledger.record_signal(signal)
    ledger.record_outcome(outcomes[0])

    records = ledger.records()

    assert len(records) == 2
    assert records[0]["type"] == "SIGNAL"
    assert records[1]["type"] == "OUTCOME"
    assert records[1]["horizon_minutes"] == 5
    assert records[1]["entry_price"] == "100.45"
    assert records[1]["exit_price"] == "100.80"


def test_ledger_is_append_only(tmp_path) -> None:
    ledger = IntradayResearchLedger(
        tmp_path / "intraday.jsonl"
    )

    signal = candidate()

    ledger.record_signal(signal)
    ledger.record_signal(signal)

    records = ledger.records()

    assert len(records) == 2


def test_persistence_has_no_execution_dependency() -> None:
    import finance.intraday.persistence as persistence

    source = open(
        persistence.__file__,
        encoding="utf-8",
    ).read()

    assert "finance.execution" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source


def test_ledger_records_strategy_version(tmp_path) -> None:
    ledger = IntradayResearchLedger(
        tmp_path / "intraday.jsonl",
        strategy_version="intraday-momentum-v1-test",
    )

    signal = candidate()
    ledger.record_signal(signal)

    outcome = IntradaySignalEvaluator(
        horizons_minutes=(1,)
    ).evaluate(
        signal,
        [
            obs(6, "100.55"),
        ],
    )[0]

    ledger.record_outcome(outcome)

    records = ledger.records()

    assert len(records) == 2
    assert records[0]["type"] == "SIGNAL"
    assert records[1]["type"] == "OUTCOME"

    assert (
        records[0]["strategy_version"]
        == "intraday-momentum-v1-test"
    )
    assert (
        records[1]["strategy_version"]
        == "intraday-momentum-v1-test"
    )
