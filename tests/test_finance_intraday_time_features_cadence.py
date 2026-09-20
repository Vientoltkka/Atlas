from datetime import datetime, timedelta, timezone
from decimal import Decimal

from finance.intraday.models import IntradayObservation
from finance.intraday.time_features import (
    TimeBasedIntradayFeatureEngine,
)


BASE = datetime(
    2026, 9, 20, 12, 0, tzinfo=timezone.utc
)


def obs(seconds, price):
    value = Decimal(str(price))
    return IntradayObservation(
        symbol="BTC-USD",
        price=value,
        bid=value - Decimal("0.05"),
        ask=value + Decimal("0.05"),
        timestamp=BASE + timedelta(seconds=seconds),
    )


def test_all_features_are_stable_with_extra_raw_samples():
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=1,
        long_minutes=5,
        volatility_step_seconds=60,
    )

    sparse = [
        obs(0, 100),
        obs(60, 101),
        obs(120, 102),
        obs(180, 103),
        obs(240, 104),
        obs(300, 105),
        obs(360, 106),
    ]

    dense = [
        obs(0, 100),
        obs(30, 100.4),
        obs(60, 101),
        obs(90, 101.4),
        obs(120, 102),
        obs(150, 102.4),
        obs(180, 103),
        obs(210, 103.4),
        obs(240, 104),
        obs(270, 104.4),
        obs(300, 105),
        obs(330, 105.4),
        obs(360, 106),
    ]

    sparse_result = engine.calculate(sparse)
    dense_result = engine.calculate(dense)

    assert sparse_result.return_short == dense_result.return_short
    assert sparse_result.return_long == dense_result.return_long
    assert sparse_result.acceleration == dense_result.acceleration
    assert (
        sparse_result.realized_volatility
        == dense_result.realized_volatility
    )


def test_acceleration_uses_exact_elapsed_time_anchor():
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=1,
        long_minutes=3,
        maximum_reference_lateness_seconds=30,
        volatility_step_seconds=60,
    )

    observations = [
        obs(200, 100),
        obs(240, 101),
        obs(250, 102),
        obs(300, 104),
        obs(380, 107),
    ]

    result = engine.calculate(observations)

    current_short = (
        Decimal("107") / Decimal("104")
        - Decimal("1")
    )

    previous_short = (
        Decimal("104") / Decimal("102")
        - Decimal("1")
    )

    assert result.acceleration == (
        current_short - previous_short
    )
