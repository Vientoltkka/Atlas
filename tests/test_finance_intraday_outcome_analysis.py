from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.intraday.evaluation import IndependentSignalEvent
from finance.intraday.models import IntradayObservation
from finance.intraday.outcome_analysis import (
    HORIZONS,
    aggregate_event_report,
)


START = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def observation(minute: int, price: str) -> IntradayObservation:
    value = Decimal(price)
    return IntradayObservation(
        symbol="BTC-USD",
        price=value,
        bid=value - Decimal("0.01"),
        ask=value + Decimal("0.01"),
        timestamp=START + timedelta(minutes=minute),
    )


def event(
    *,
    direction: str = "LONG",
    signal: str = "MOMENTUM",
    source: str = "alpha",
    cost: str = "0.001",
) -> IndependentSignalEvent:
    return IndependentSignalEvent(
        "BTC-USD",
        direction,
        signal,
        source,
        START,
        Decimal("100"),
        estimated_round_trip_cost=Decimal(cost),
    )


def test_report_aggregates_long_short_events_and_horizons() -> None:
    long_event = event()
    short_event = event(direction="SHORT")
    future = [
        observation(1, "101"),
        observation(5, "102"),
        observation(15, "99"),
        observation(30, "104"),
    ]
    long_event.calculate_outcomes(future)
    short_event.calculate_outcomes(future)

    report = aggregate_event_report([short_event, long_event])

    assert report.total_events == 2
    assert report.horizons[1].outcomes_available == 2
    assert report.horizons[1].mean_gross_return == Decimal("0")
    assert report.horizons[5].mean_gross_return == Decimal("0")
    assert report.horizons[15].mean_gross_return == Decimal("0")
    assert report.horizons[30].mean_gross_return == Decimal("0")
    assert report.horizons[15].net_win_rate == Decimal("0.5")


def test_net_mean_differs_from_gross_mean_when_cost_is_present() -> None:
    item = event(cost="0.003")
    item.calculate_outcomes([observation(1, "101")])

    horizon = aggregate_event_report([item]).horizons[1]

    assert horizon.mean_gross_return == Decimal("0.01")
    assert horizon.mean_net_return == Decimal("0.007")


def test_missing_outcomes_are_excluded_from_metrics() -> None:
    complete = event()
    complete.calculate_outcomes([observation(1, "101")])
    missing = event(signal="BREAKOUT")

    horizon = aggregate_event_report([complete, missing]).horizons[1]

    assert horizon.outcomes_available == 1
    assert horizon.outcomes_pending == 1
    assert horizon.mean_gross_return == Decimal("0.01")
    assert horizon.mean_net_return == Decimal("0.009")
    assert horizon.net_win_rate == Decimal("1")


def test_horizon_without_data_has_none_metrics() -> None:
    item = event()
    item.calculate_outcomes([observation(1, "101")])

    horizon = aggregate_event_report([item]).horizons[5]

    assert horizon.outcomes_available == 0
    assert horizon.outcomes_pending == 1
    assert horizon.mean_gross_return is None
    assert horizon.mean_net_return is None
    assert horizon.net_win_rate is None


def test_signal_source_groups_are_deterministically_sorted() -> None:
    events = [
        event(signal="Z", source="beta"),
        event(signal="A", source="zeta"),
        event(signal="A", source="alpha"),
        event(signal="A", source="alpha"),
    ]

    groups = aggregate_event_report(events).events_by_signal_source

    assert [(item.signal, item.source, item.event_count) for item in groups] == [
        ("A", "alpha", 2),
        ("A", "zeta", 1),
        ("Z", "beta", 1),
    ]


def test_report_module_has_no_execution_or_broker_dependency() -> None:
    import finance.intraday.outcome_analysis as outcome_analysis

    source = open(outcome_analysis.__file__, encoding="utf-8").read()
    assert "ExecutionIntent" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source
    assert "finance.execution" not in source
    assert HORIZONS == (1, 5, 15, 30)
