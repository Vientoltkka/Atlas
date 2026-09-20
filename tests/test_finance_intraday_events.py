from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.intraday.evaluation import (
    DEFAULT_ESTIMATED_ROUND_TRIP_COST,
    EventDetector,
    IndependentSignalEvent,
)
from finance.intraday.models import IntradayObservation


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


def detect(detector: EventDetector, minute: int, **kwargs):
    return detector.observe(
        symbol=kwargs.get("symbol", "BTC-USD"),
        direction=kwargs.get("direction", "LONG"),
        signal=kwargs.get("signal", "MOMENTUM"),
        source=kwargs.get("source", "test"),
        timestamp=START + timedelta(minutes=minute),
        entry_price=Decimal(kwargs.get("entry_price", "100")),
    )


def test_twenty_equivalent_observations_are_one_event() -> None:
    detector = EventDetector()
    events = [detect(detector, minute) for minute in range(20)]

    assert len({event.id for event in events}) == 1
    assert events[0].observation_count == 20
    assert events[-1] is events[0]


def test_event_is_new_when_cooldown_expires() -> None:
    detector = EventDetector(cooldown_minutes=15)

    first = detect(detector, 0)
    repeated = detect(detector, 14)
    new = detect(detector, 30)

    assert repeated is first
    assert new.id != first.id
    assert new.observation_count == 1


def test_symbol_direction_and_signal_source_are_independent_keys() -> None:
    detector = EventDetector()
    first = detect(detector, 0)

    assert detect(detector, 1, symbol="ETH-USD").id != first.id
    assert detect(detector, 1, direction="SHORT").id != first.id
    assert detect(detector, 1, signal="MEAN_REVERSION").id != first.id
    assert detect(detector, 1, source="other-engine").id != first.id


def test_long_and_short_gross_returns_are_decimal() -> None:
    long_event = IndependentSignalEvent(
        "BTC-USD", "LONG", "MOMENTUM", "test", START, Decimal("100"),
    )
    short_event = IndependentSignalEvent(
        "BTC-USD", "SHORT", "MOMENTUM", "test", START, Decimal("100"),
    )
    future = [observation(1, "101"), observation(5, "98")]

    long_event.calculate_outcomes(future)
    short_event.calculate_outcomes(future)

    assert long_event.outcome_1m.gross_return == Decimal("0.01")
    assert short_event.outcome_1m.gross_return == Decimal("-0.01")
    assert short_event.outcome_5m.gross_return == Decimal("0.02")


def test_cost_is_subtracted_from_net_return() -> None:
    event = IndependentSignalEvent(
        "BTC-USD", "LONG", "MOMENTUM", "test", START, Decimal("100"),
        estimated_round_trip_cost=Decimal("0.003"),
    )

    event.calculate_outcomes([observation(1, "101")])

    assert event.outcome_1m.net_return == Decimal("0.007")


def test_missing_future_prices_are_none_not_zero() -> None:
    event = IndependentSignalEvent(
        "BTC-USD", "LONG", "MOMENTUM", "test", START, Decimal("100"),
    )

    event.calculate_outcomes([observation(1, "101")])

    assert event.outcome_5m.gross_return is None
    assert event.outcome_5m.net_return is None
    assert event.outcome_5m.future_price is None


def test_too_late_future_price_is_not_an_outcome() -> None:
    event = IndependentSignalEvent(
        "BTC-USD", "LONG", "MOMENTUM", "test", START, Decimal("100"),
    )

    event.calculate_outcomes([observation(6, "102")])

    assert event.outcome_5m.gross_return is None
    assert event.outcome_5m.net_return is None
    assert event.outcome_5m.future_price is None


def test_detector_rejects_invalid_direction() -> None:
    detector = EventDetector()

    with pytest.raises(ValueError, match="direction"):
        detect(detector, 0, direction="SIDEWAYS")


def test_detector_rejects_naive_timestamp() -> None:
    detector = EventDetector()

    with pytest.raises(ValueError, match="timezone-aware"):
        detector.observe(
            symbol="BTC-USD",
            direction="LONG",
            signal="MOMENTUM",
            source="test",
            timestamp=datetime(2026, 9, 19, 12, 0),
            entry_price=Decimal("100"),
        )


def test_detector_rejects_out_of_order_observation() -> None:
    detector = EventDetector()
    detect(detector, 10)
    detect(detector, 30)

    with pytest.raises(ValueError, match="move backwards"):
        detect(detector, 20)


def test_cooldown_and_cost_are_configurable() -> None:
    detector = EventDetector(
        cooldown_minutes=3,
        estimated_round_trip_cost=Decimal("0.005"),
    )

    first = detect(detector, 0)
    second = detect(detector, 3)

    assert second.id != first.id
    assert second.cooldown_minutes == 3
    assert second.estimated_round_trip_cost == Decimal("0.005")
    assert DEFAULT_ESTIMATED_ROUND_TRIP_COST == Decimal("0.0010")


def test_event_module_has_no_execution_or_broker_dependency() -> None:
    import finance.intraday.evaluation as evaluation

    source = open(evaluation.__file__, encoding="utf-8").read()
    assert "ExecutionIntent" not in source
    assert "BrokerAdapter" not in source
    assert "PaperFinanceService" not in source
