from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.intraday.features import IntradayFeatureEngine
from finance.intraday.models import (
    IntradayObservation,
    IntradaySignalAction,
)
from finance.intraday.service import IntradaySignalService
from finance.intraday.signal_engine import (
    IntradaySignalEngine,
    IntradaySignalPolicy,
)


START = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def obs(
    minute: int,
    price: str,
    *,
    spread: str = "0.10",
) -> IntradayObservation:
    value = Decimal(price)
    half = Decimal(spread) / Decimal("2")

    return IntradayObservation(
        symbol="BTC-USD",
        price=value,
        bid=value - half,
        ask=value + half,
        timestamp=START + timedelta(minutes=minute),
    )


def test_observation_requires_aware_timestamp() -> None:
    with pytest.raises(ValueError):
        IntradayObservation(
            symbol="BTC-USD",
            price=Decimal("100"),
            bid=Decimal("99"),
            ask=Decimal("101"),
            timestamp=START.replace(tzinfo=None),
        )


def test_feature_engine_requires_enough_observations() -> None:
    engine = IntradayFeatureEngine()

    with pytest.raises(ValueError):
        engine.calculate(
            [
                obs(0, "100"),
                obs(1, "101"),
            ]
        )


def test_features_detect_positive_momentum() -> None:
    engine = IntradayFeatureEngine()

    observations = [
        obs(0, "100.00"),
        obs(1, "100.05"),
        obs(2, "100.10"),
        obs(3, "100.15"),
        obs(4, "100.25"),
        obs(5, "100.45"),
    ]

    features = engine.calculate(observations)

    assert features.return_short > 0
    assert features.return_long > 0
    assert features.acceleration > 0
    assert features.relative_spread > 0


def test_signal_candidate_when_all_policy_conditions_match() -> None:
    features = IntradayFeatureEngine().calculate(
        [
            obs(0, "100.00"),
            obs(1, "100.02"),
            obs(2, "100.04"),
            obs(3, "100.08"),
            obs(4, "100.20"),
            obs(5, "100.45"),
        ]
    )

    policy = IntradaySignalPolicy(
        minimum_short_return=Decimal("0.0010"),
        minimum_long_return=Decimal("0.0020"),
        minimum_acceleration=Decimal("0"),
        maximum_volatility=Decimal("0.01"),
        maximum_relative_spread=Decimal("0.003"),
    )

    signal = IntradaySignalEngine(policy).evaluate(features)

    assert signal.action is IntradaySignalAction.CANDIDATE
    assert signal.reasons == ("POLICY_MATCH",)


def test_signal_no_action_on_flat_market() -> None:
    features = IntradayFeatureEngine().calculate(
        [
            obs(0, "100"),
            obs(1, "100"),
            obs(2, "100"),
            obs(3, "100"),
            obs(4, "100"),
            obs(5, "100"),
        ]
    )

    signal = IntradaySignalEngine().evaluate(features)

    assert signal.action is IntradaySignalAction.NO_ACTION
    assert "SHORT_MOMENTUM_TOO_LOW" in signal.reasons
    assert "LONG_MOMENTUM_TOO_LOW" in signal.reasons


def test_signal_rejects_wide_spread() -> None:
    features = IntradayFeatureEngine().calculate(
        [
            obs(0, "100"),
            obs(1, "100.05"),
            obs(2, "100.10"),
            obs(3, "100.20"),
            obs(4, "100.35"),
            obs(5, "100.60", spread="1.00"),
        ]
    )

    signal = IntradaySignalEngine().evaluate(features)

    assert signal.action is IntradaySignalAction.NO_ACTION
    assert "SPREAD_TOO_WIDE" in signal.reasons


def test_service_emits_only_after_minimum_window() -> None:
    service = IntradaySignalService()

    observations = [
        obs(0, "100"),
        obs(1, "100.02"),
        obs(2, "100.04"),
        obs(3, "100.08"),
        obs(4, "100.20"),
        obs(5, "100.45"),
    ]

    for item in observations[:-1]:
        assert service.ingest(item) is None

    assert service.ingest(observations[-1]) is not None


def test_service_rejects_duplicate_or_old_timestamp() -> None:
    service = IntradaySignalService()

    first = obs(0, "100")

    assert service.ingest(first) is None
    assert service.ingest(first) is None


def test_intraday_layer_has_no_execution_dependency() -> None:
    import finance.intraday.features as features
    import finance.intraday.service as service
    import finance.intraday.signal_engine as signal_engine

    for module in (features, service, signal_engine):
        source = open(module.__file__, encoding="utf-8").read()

        assert "finance.execution" not in source
        assert "BrokerAdapter" not in source
        assert "ExecutionIntent" not in source
        assert "PaperFinanceService" not in source
