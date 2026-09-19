"""Outcome analysis for time-based intraday research.

Research only:
- no broker access
- no orders
- no execution
- no parameter optimization

Measures forward returns after historical momentum states.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from statistics import mean, median

from finance.intraday.models import IntradayObservation
from finance.intraday.time_features import (
    TimeBasedIntradayFeatureEngine,
)
from finance.intraday.time_replay import (
    ReplayConfiguration,
    load_observations,
)


@dataclass(frozen=True)
class MomentumOutcome:
    timestamp: object
    return_short: Decimal
    return_long: Decimal
    acceleration: Decimal
    forward_1m: Decimal | None
    forward_5m: Decimal | None
    forward_15m: Decimal | None
    forward_30m: Decimal | None


HORIZONS = (1, 5, 15, 30)


def percentile(values, q):
    ordered = sorted(values)
    index = int((len(ordered) - 1) * q)
    return ordered[index]


def future_return(
    observations: list[IntradayObservation],
    *,
    index: int,
    minutes: int,
    maximum_lateness_seconds: int = 30,
) -> Decimal | None:
    current = observations[index]
    target = current.timestamp + timedelta(minutes=minutes)
    maximum = target + timedelta(seconds=maximum_lateness_seconds)

    for item in observations[index + 1:]:
        if item.symbol != current.symbol:
            continue

        if item.timestamp < target:
            continue

        if item.timestamp > maximum:
            return None

        return (item.price / current.price) - Decimal("1")

    return None


def build_outcomes(
    observations: list[IntradayObservation],
    configuration: ReplayConfiguration,
) -> list[MomentumOutcome]:
    engine = TimeBasedIntradayFeatureEngine(
        short_minutes=configuration.short_minutes,
        long_minutes=configuration.long_minutes,
    )

    outcomes = []

    for index in range(1, len(observations)):
        history = observations[: index + 1]

        try:
            features = engine.calculate(history)
        except ValueError:
            continue

        forward = {
            horizon: future_return(
                observations,
                index=index,
                minutes=horizon,
            )
            for horizon in HORIZONS
        }

        outcomes.append(
            MomentumOutcome(
                timestamp=features.timestamp,
                return_short=features.return_short,
                return_long=features.return_long,
                acceleration=features.acceleration,
                forward_1m=forward[1],
                forward_5m=forward[5],
                forward_15m=forward[15],
                forward_30m=forward[30],
            )
        )

    return outcomes


def report_forward(label, rows):
    print()
    print(label)
    print("-" * len(label))
    print(f"States: {len(rows)}")

    for horizon in HORIZONS:
        attribute = f"forward_{horizon}m"

        values = [
            getattr(row, attribute)
            for row in rows
            if getattr(row, attribute) is not None
        ]

        if not values:
            print(f"+{horizon:>2}m: no complete outcomes")
            continue

        floats = [float(value) for value in values]

        positive = sum(
            value > Decimal("0")
            for value in values
        )

        print(
            f"+{horizon:>2}m: "
            f"n={len(values):4d} "
            f"mean={mean(floats): .6%} "
            f"median={median(floats): .6%} "
            f"positive={100 * positive / len(values):6.2f}%"
        )


def analyze(observations, configuration):
    print()
    print("=" * 72)
    print(
        "WINDOW "
        f"{configuration.short_minutes}m/"
        f"{configuration.long_minutes}m"
    )
    print("=" * 72)

    rows = build_outcomes(
        observations,
        configuration,
    )

    print(f"Evaluable states: {len(rows)}")

    if not rows:
        return

    short_values = [row.return_short for row in rows]
    long_values = [row.return_long for row in rows]

    short_p90 = percentile(short_values, .90)
    short_p95 = percentile(short_values, .95)

    long_p90 = percentile(long_values, .90)
    long_p95 = percentile(long_values, .95)

    print(
        f"SHORT p90={float(short_p90):.6%} "
        f"p95={float(short_p95):.6%}"
    )

    print(
        f"LONG  p90={float(long_p90):.6%} "
        f"p95={float(long_p95):.6%}"
    )

    report_forward(
        "ALL EVALUABLE STATES",
        rows,
    )

    positive_momentum = [
        row
        for row in rows
        if (
            row.return_short > 0
            and row.return_long > 0
            and row.acceleration > 0
        )
    ]

    report_forward(
        "POSITIVE MOMENTUM",
        positive_momentum,
    )

    p90_momentum = [
        row
        for row in rows
        if (
            row.return_short >= short_p90
            and row.return_long >= long_p90
            and row.acceleration > 0
        )
    ]

    report_forward(
        "P90 SHORT + P90 LONG + POSITIVE ACCELERATION",
        p90_momentum,
    )

    p95_momentum = [
        row
        for row in rows
        if (
            row.return_short >= short_p95
            and row.return_long >= long_p95
            and row.acceleration > 0
        )
    ]

    report_forward(
        "P95 SHORT + P95 LONG + POSITIVE ACCELERATION",
        p95_momentum,
    )


def main():
    from pathlib import Path

    path = Path(
        ".atlas/finance_intraday/live_research.jsonl"
    )

    observations = load_observations(path)

    print("=== ATLAS INTRADAY OUTCOME ANALYZER V2 ===")
    print("RESEARCH ONLY - NO EXECUTION")
    print(f"Observations: {len(observations)}")

    configurations = (
        ReplayConfiguration(1, 3),
        ReplayConfiguration(1, 5),
        ReplayConfiguration(3, 5),
    )

    for configuration in configurations:
        analyze(
            observations,
            configuration,
        )


if __name__ == "__main__":
    main()
