from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json

import pytest

from finance.intraday.models import IntradayObservation
from finance.intraday.outcome_analysis import (
    build_outcomes,
    future_return,
)
from finance.intraday.time_replay import (
    ReplayConfiguration,
    TIME_BASED_STRATEGY_VERSION,
    load_observations,
)


BASE = datetime(
    2026, 9, 19, 12, 0, tzinfo=timezone.utc
)


def obs(minutes, price):
    value = Decimal(str(price))

    return IntradayObservation(
        symbol="BTC-USD",
        price=value,
        bid=value - Decimal("0.05"),
        ask=value + Decimal("0.05"),
        timestamp=BASE + timedelta(minutes=minutes),
    )


def test_future_return_uses_time_horizon():
    observations = [
        obs(0, 100),
        obs(1, 101),
        obs(5, 105),
    ]

    result = future_return(
        observations,
        index=0,
        minutes=5,
    )

    assert result == Decimal("0.05")


def test_time_replay_builds_forward_outcomes():
    observations = [
        obs(0, 100),
        obs(1, 101),
        obs(2, 102),
        obs(3, 103),
        obs(4, 104),
        obs(5, 105),
        obs(6, 106),
        obs(7, 107),
        obs(8, 108),
        obs(9, 109),
        obs(10, 110),
    ]

    rows = build_outcomes(
        observations,
        ReplayConfiguration(
            short_minutes=1,
            long_minutes=3,
        ),
    )

    assert rows
    assert rows[-1].return_short > 0
    assert rows[-1].return_long > 0


def test_replay_recovers_complete_observations_before_truncated_final_line(
    tmp_path,
):
    path = tmp_path / "truncated-replay.jsonl"
    records = [
        {
            "type": "OBSERVATION",
            "strategy_version": TIME_BASED_STRATEGY_VERSION,
            "capture_id": "capture-a",
            "symbol": "BTC-USD",
            "price": str(price),
            "bid": str(price - Decimal("0.05")),
            "ask": str(price + Decimal("0.05")),
            "timestamp": (BASE + timedelta(minutes=index)).isoformat(),
        }
        for index, price in enumerate((100, 101))
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records)
        + '\n{"type":"OBSERVATION"',
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="ignored invalid final non-empty line"):
        observations = load_observations(path, capture_id="capture-a")

    assert [item.price for item in observations] == [Decimal("100"), Decimal("101")]


def test_replay_rejects_corrupt_intermediate_line(tmp_path):
    path = tmp_path / "corrupt-replay.jsonl"
    path.write_text(
        '{"type":"OBSERVATION"}\n{broken\n{"type":"OBSERVATION"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSONL at line 2"):
        load_observations(path)
