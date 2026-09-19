"""Forward evaluation of intraday candidate signals.

Evaluation is research-only and has no order-execution capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from finance.intraday.models import (
    IntradayObservation,
    IntradaySignal,
    IntradaySignalAction,
)


@dataclass(frozen=True)
class IntradaySignalOutcome:
    symbol: str
    signal_timestamp: datetime
    horizon_minutes: int
    entry_price: Decimal
    exit_price: Decimal
    forward_return: Decimal
    maximum_favorable_excursion: Decimal
    maximum_adverse_excursion: Decimal
    observations: int


class IntradaySignalEvaluator:
    def __init__(
        self,
        *,
        horizons_minutes: tuple[int, ...] = (1, 5, 15, 30),
        maximum_lateness_seconds: int = 30,
    ) -> None:
        if not horizons_minutes:
            raise ValueError("at least one horizon is required")

        if any(value <= 0 for value in horizons_minutes):
            raise ValueError("horizons must be positive")

        if len(set(horizons_minutes)) != len(horizons_minutes):
            raise ValueError("horizons must be unique")

        if maximum_lateness_seconds < 0:
            raise ValueError("maximum lateness cannot be negative")

        self._horizons = tuple(sorted(horizons_minutes))
        self._maximum_lateness = timedelta(
            seconds=maximum_lateness_seconds
        )

    def evaluate(
        self,
        signal: IntradaySignal,
        observations: list[IntradayObservation],
    ) -> tuple[IntradaySignalOutcome, ...]:
        if signal.action is not IntradaySignalAction.CANDIDATE:
            return ()

        future = [
            item
            for item in observations
            if item.symbol == signal.symbol
            and item.timestamp > signal.timestamp
        ]

        if not future:
            return ()

        future.sort(key=lambda item: item.timestamp)

        entry = signal.features.last_price
        outcomes: list[IntradaySignalOutcome] = []

        for horizon in self._horizons:
            target = signal.timestamp + timedelta(minutes=horizon)

            exit_observation = next(
                (
                    item
                    for item in future
                    if item.timestamp >= target
                ),
                None,
            )

            # Do not score an incomplete or excessively late horizon.
            if exit_observation is None:
                continue

            if (
                exit_observation.timestamp - target
                > self._maximum_lateness
            ):
                continue

            eligible = [
                item
                for item in future
                if item.timestamp <= exit_observation.timestamp
            ]

            prices = [item.price for item in eligible]
            exit_price = exit_observation.price

            forward_return = (exit_price / entry) - Decimal("1")
            mfe = (max(prices) / entry) - Decimal("1")
            mae = (min(prices) / entry) - Decimal("1")

            outcomes.append(
                IntradaySignalOutcome(
                    symbol=signal.symbol,
                    signal_timestamp=signal.timestamp,
                    horizon_minutes=horizon,
                    entry_price=entry,
                    exit_price=exit_price,
                    forward_return=forward_return,
                    maximum_favorable_excursion=mfe,
                    maximum_adverse_excursion=mae,
                    observations=len(eligible),
                )
            )

        return tuple(outcomes)
