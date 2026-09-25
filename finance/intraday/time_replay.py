"""Replay real observations through time-based intraday features.

Research-only. No orders, no broker access, no execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

TIME_BASED_STRATEGY_VERSION = "intraday-time-research-v2"

from finance.intraday.models import IntradayObservation
from finance.intraday.persistence import read_jsonl_records
from finance.intraday.time_features import (
    TimeBasedIntradayFeatureEngine,
)


@dataclass(frozen=True)
class ReplayConfiguration:
    short_minutes: int
    long_minutes: int


CONFIGURATIONS = (
    ReplayConfiguration(1, 3),
    ReplayConfiguration(1, 5),
    ReplayConfiguration(3, 5),
)


def load_observations(
    path: Path,
    *,
    strategy_version: str = TIME_BASED_STRATEGY_VERSION,
    capture_id: str | None = None,
) -> list[IntradayObservation]:
    observations = []
    records = read_jsonl_records(path)
    version_records = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("strategy_version") == strategy_version
    ]
    capture_ids = sorted({
        record["capture_id"]
        for record in version_records
        if isinstance(record.get("capture_id"), str)
        and record["capture_id"].strip()
    })
    if capture_id is None and len(capture_ids) > 1:
        raise ValueError(
            "capture_id must be selected; available capture IDs: "
            + ", ".join(capture_ids)
        )
    if capture_id is None and len(capture_ids) == 1:
        capture_id = capture_ids[0]
    if capture_id is not None and capture_id not in capture_ids:
        raise ValueError(
            f"unknown capture_id for strategy_version {strategy_version}: "
            f"{capture_id}"
        )

    for record in records:

        if record.get("type") != "OBSERVATION":
            continue
        if record.get("strategy_version") != strategy_version:
            continue
        if capture_id is not None and record.get("capture_id") != capture_id:
            continue
        if capture_id is None and record.get("capture_id") is not None:
            continue

        observations.append(
            IntradayObservation(
                symbol=record["symbol"],
                price=Decimal(str(record["price"])),
                bid=Decimal(str(record["bid"])),
                ask=Decimal(str(record["ask"])),
                timestamp=datetime.fromisoformat(record["timestamp"]),
                volume=(
                    Decimal(str(record["volume"]))
                    if record.get("volume") is not None
                    else None
                ),
            )
        )

    observations.sort(key=lambda item: item.timestamp)
    return observations


def percentile(values, q):
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


def describe(name, values):
    values = [float(item) for item in values]

    print(name)
    print(f"  min : {min(values): .8f}")
    print(f"  p50 : {median(values): .8f}")
    print(f"  p90 : {percentile(values, .90): .8f}")
    print(f"  p95 : {percentile(values, .95): .8f}")
    print(f"  p99 : {percentile(values, .99): .8f}")
    print(f"  max : {max(values): .8f}")
    print(f"  mean: {mean(values): .8f}")


def replay(observations, configuration):
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=configuration.short_minutes,
        long_minutes=configuration.long_minutes,
    )

    features = []
    rejected = 0

    for index in range(1, len(observations)):
        history = observations[: index + 1]

        try:
            result = engine.calculate(history)
        except ValueError:
            rejected += 1
            continue

        features.append(result)

    return features, rejected


def main():
    path = Path(
        ".atlas/finance_intraday/live_research.jsonl"
    )

    observations = load_observations(path)

    print("=== ATLAS TIME-BASED REPLAY V2 ===")
    print("RESEARCH ONLY - NO EXECUTION")
    print(f"Observations: {len(observations)}")

    if observations:
        print(f"First: {observations[0].timestamp.isoformat()}")
        print(f"Last : {observations[-1].timestamp.isoformat()}")

    for configuration in CONFIGURATIONS:
        print()
        print("=" * 64)
        print(
            "WINDOW "
            f"{configuration.short_minutes}m/"
            f"{configuration.long_minutes}m"
        )
        print("=" * 64)

        features, rejected = replay(
            observations,
            configuration,
        )

        print(f"Evaluable: {len(features)}")
        print(f"Rejected : {rejected}")

        if not features:
            continue

        describe(
            "SHORT RETURN",
            [item.return_short for item in features],
        )

        describe(
            "LONG RETURN",
            [item.return_long for item in features],
        )

        describe(
            "ACCELERATION",
            [item.acceleration for item in features],
        )

        describe(
            "REALIZED VOLATILITY",
            [
                item.realized_volatility
                for item in features
            ],
        )

        describe(
            "RELATIVE SPREAD",
            [
                item.relative_spread
                for item in features
            ],
        )

        current_policy_matches = [
            item
            for item in features
            if (
                item.return_short >= Decimal("0.0010")
                and item.return_long >= Decimal("0.0020")
                and item.acceleration > Decimal("0")
                and item.realized_volatility <= Decimal("0.0100")
                and item.relative_spread <= Decimal("0.0030")
            )
        ]

        print()
        print(
            "Current V1 thresholds would match: "
            f"{len(current_policy_matches)}/{len(features)} "
            f"({100 * len(current_policy_matches) / len(features):.4f}%)"
        )


if __name__ == "__main__":
    main()

