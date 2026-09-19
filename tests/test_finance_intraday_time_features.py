from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from finance.intraday.models import IntradayObservation
from finance.intraday.time_features import (
    TimeBasedIntradayFeatureEngine,
)


BASE = datetime(
    2026,
    9,
    19,
    12,
    0,
    tzinfo=timezone.utc,
)


def obs(seconds, price):
    price = Decimal(str(price))

    return IntradayObservation(
        symbol="BTC-USD",
        price=price,
        bid=price - Decimal("0.05"),
        ask=price + Decimal("0.05"),
        timestamp=BASE + timedelta(seconds=seconds),
    )


def test_time_features_use_elapsed_time_not_count():
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=1,
        long_minutes=5,
    )

    observations = [
        obs(0, 100),
        obs(30, 101),
        obs(60, 102),
        obs(120, 103),
        obs(180, 104),
        obs(240, 105),
        obs(300, 106),
        obs(360, 107),
    ]

    result = engine.calculate(observations)

    assert result.return_short == (
        Decimal("107") / Decimal("106") - Decimal("1")
    )

    assert result.return_long == (
        Decimal("107") / Decimal("102") - Decimal("1")
    )


def test_time_features_are_stable_with_extra_samples():
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=1,
        long_minutes=5,
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
        obs(30, 100.5),
        obs(60, 101),
        obs(90, 101.5),
        obs(120, 102),
        obs(150, 102.5),
        obs(180, 103),
        obs(210, 103.5),
        obs(240, 104),
        obs(270, 104.5),
        obs(300, 105),
        obs(330, 105.5),
        obs(360, 106),
    ]

    sparse_result = engine.calculate(sparse)
    dense_result = engine.calculate(dense)

    assert sparse_result.return_short == dense_result.return_short
    assert sparse_result.return_long == dense_result.return_long


def test_time_features_require_history():
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=1,
        long_minutes=5,
    )

    with pytest.raises(
        ValueError,
        match="insufficient time history",
    ):
        engine.calculate(
            [
                obs(0, 100),
                obs(60, 101),
            ]
        )


def test_time_features_reject_stale_reference():
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=1,
        long_minutes=5,
        maximum_reference_lateness_seconds=10,
    )

    observations = [
        obs(0, 100),
        obs(60, 101),
        obs(120, 102),
        obs(180, 103),
        obs(240, 104),
        obs(300, 105),
        obs(380, 106),
    ]

    with pytest.raises(
        ValueError,
        match="reference observation too old",
    ):
        engine.calculate(observations)

