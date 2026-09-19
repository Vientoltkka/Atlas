from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.intraday.bridge import observation_from_quote
from finance.intraday.evaluation import IntradaySignalEvaluator
from finance.intraday.features import IntradayFeatureEngine
from finance.intraday.models import (
    IntradayObservation,
    IntradaySignalAction,
)
from finance.intraday.signal_engine import IntradaySignalEngine
from finance.market_data.models import Quote


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


def candidate_signal():
    observations = [
        obs(0, "100.00"),
        obs(1, "100.02"),
        obs(2, "100.04"),
        obs(3, "100.08"),
        obs(4, "100.20"),
        obs(5, "100.45"),
    ]

    features = IntradayFeatureEngine().calculate(observations)
    signal = IntradaySignalEngine().evaluate(features)

    assert signal.action is IntradaySignalAction.CANDIDATE
    return signal


def test_quote_bridge_uses_midpoint_and_preserves_timestamp() -> None:
    quote = Quote(
        symbol="BTC-USD",
        bid=Decimal("81711"),
        ask=Decimal("81713"),
        timestamp=START,
        provider="TEST",
    )

    observation = observation_from_quote(quote)

    assert observation.symbol == "BTC-USD"
    assert observation.price == Decimal("81712")
    assert observation.bid == quote.bid
    assert observation.ask == quote.ask
    assert observation.timestamp == quote.timestamp


def test_evaluator_scores_completed_horizons_without_lookahead() -> None:
    signal = candidate_signal()

    future = [
        obs(6, "100.55"),
        obs(7, "100.40"),
        obs(8, "100.70"),
        obs(9, "100.60"),
        obs(10, "100.80"),
    ]

    outcomes = IntradaySignalEvaluator(
        horizons_minutes=(1, 5)
    ).evaluate(signal, future)

    assert len(outcomes) == 2

    one_minute = outcomes[0]
    five_minutes = outcomes[1]

    assert one_minute.horizon_minutes == 1
    assert one_minute.exit_price == Decimal("100.55")
    assert one_minute.observations == 1

    assert five_minutes.horizon_minutes == 5
    assert five_minutes.exit_price == Decimal("100.80")
    assert five_minutes.observations == 5

    assert five_minutes.maximum_favorable_excursion > 0
    assert (
        five_minutes.maximum_adverse_excursion
        < five_minutes.maximum_favorable_excursion
    )


def test_evaluator_accepts_first_observation_after_target_within_tolerance() -> None:
    signal = candidate_signal()

    target = signal.timestamp + timedelta(minutes=5)

    late = IntradayObservation(
        symbol="BTC-USD",
        price=Decimal("100.90"),
        bid=Decimal("100.85"),
        ask=Decimal("100.95"),
        timestamp=target + timedelta(seconds=7),
    )

    outcomes = IntradaySignalEvaluator(
        horizons_minutes=(5,),
        maximum_lateness_seconds=30,
    ).evaluate(signal, [late])

    assert len(outcomes) == 1
    assert outcomes[0].exit_price == Decimal("100.90")
    assert outcomes[0].observations == 1


def test_evaluator_rejects_observation_beyond_lateness_tolerance() -> None:
    signal = candidate_signal()

    target = signal.timestamp + timedelta(minutes=5)

    too_late = IntradayObservation(
        symbol="BTC-USD",
        price=Decimal("100.90"),
        bid=Decimal("100.85"),
        ask=Decimal("100.95"),
        timestamp=target + timedelta(seconds=45),
    )

    outcomes = IntradaySignalEvaluator(
        horizons_minutes=(5,),
        maximum_lateness_seconds=30,
    ).evaluate(signal, [too_late])

    assert outcomes == ()



def test_evaluator_does_not_score_incomplete_horizon() -> None:
    signal = candidate_signal()

    future = [
        obs(6, "100.55"),
        obs(7, "100.60"),
    ]

    outcomes = IntradaySignalEvaluator(
        horizons_minutes=(5,)
    ).evaluate(signal, future)

    assert outcomes == ()


def test_evaluator_ignores_observations_before_signal() -> None:
    signal = candidate_signal()

    observations = [
        obs(1, "50"),
        obs(6, "100.55"),
    ]

    outcomes = IntradaySignalEvaluator(
        horizons_minutes=(1,)
    ).evaluate(signal, observations)

    assert len(outcomes) == 1
    assert outcomes[0].exit_price == Decimal("100.55")


def test_evaluation_has_no_execution_dependency() -> None:
    import finance.intraday.bridge as bridge
    import finance.intraday.evaluation as evaluation

    for module in (bridge, evaluation):
        source = open(
            module.__file__,
            encoding="utf-8",
        ).read()

        assert "finance.execution" not in source
        assert "ExecutionIntent" not in source
        assert "BrokerAdapter" not in source
        assert "PaperFinanceService" not in source
