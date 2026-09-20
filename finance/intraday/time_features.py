"""Time-based intraday feature calculation.

V2 research implementation.
Feature windows and volatility sampling are defined by elapsed time,
not by the number of raw market observations.
V1 remains unchanged.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal, localcontext

from finance.intraday.models import (
    IntradayFeatures,
    IntradayObservation,
)


def _return(new: Decimal, old: Decimal) -> Decimal:
    if old <= 0:
        raise ValueError("reference price must be positive")
    return (new / old) - Decimal("1")


def _sqrt(value: Decimal) -> Decimal:
    if value < 0:
        raise ValueError("cannot sqrt negative value")

    with localcontext() as ctx:
        ctx.prec = 28
        return value.sqrt()


class TimeBasedIntradayFeatureEngine:
    """Calculate cadence-independent features from elapsed-time windows."""

    def __init__(
        self,
        *,
        short_minutes: int = 1,
        long_minutes: int = 5,
        maximum_reference_lateness_seconds: int = 30,
        volatility_step_seconds: int = 60,
    ) -> None:
        if short_minutes <= 0:
            raise ValueError("short_minutes must be positive")

        if long_minutes <= short_minutes:
            raise ValueError(
                "long_minutes must be greater than short_minutes"
            )

        if maximum_reference_lateness_seconds < 0:
            raise ValueError(
                "maximum_reference_lateness_seconds cannot be negative"
            )

        if volatility_step_seconds <= 0:
            raise ValueError(
                "volatility_step_seconds must be positive"
            )

        if volatility_step_seconds > long_minutes * 60:
            raise ValueError(
                "volatility_step_seconds cannot exceed long window"
            )

        self._short = timedelta(minutes=short_minutes)
        self._long = timedelta(minutes=long_minutes)
        self._maximum_lateness = timedelta(
            seconds=maximum_reference_lateness_seconds
        )
        self._volatility_step = timedelta(
            seconds=volatility_step_seconds
        )

    @property
    def short_minutes(self) -> float:
        return self._short.total_seconds() / 60

    @property
    def long_minutes(self) -> float:
        return self._long.total_seconds() / 60

    @property
    def minimum_history(self) -> timedelta:
        return max(self._long, self._short * 2)

    def _reference(
        self,
        observations: list[IntradayObservation],
        *,
        target,
    ) -> IntradayObservation:
        eligible = [
            item
            for item in observations
            if item.timestamp <= target
        ]

        if not eligible:
            raise ValueError("insufficient time history")

        reference = eligible[-1]

        if target - reference.timestamp > self._maximum_lateness:
            raise ValueError("reference observation too old")

        return reference

    def _volatility_anchors(
        self,
        observations: list[IntradayObservation],
        *,
        last: IntradayObservation,
    ) -> list[IntradayObservation]:
        target = last.timestamp - self._long
        anchors: list[IntradayObservation] = []

        while target < last.timestamp:
            anchors.append(
                self._reference(
                    observations,
                    target=target,
                )
            )
            target += self._volatility_step

        anchors.append(last)

        for previous, current in zip(anchors, anchors[1:]):
            if current.timestamp <= previous.timestamp:
                raise ValueError(
                    "insufficient distinct observations "
                    "for volatility grid"
                )

        return anchors

    def calculate(
        self,
        observations: list[IntradayObservation],
    ) -> IntradayFeatures:
        if len(observations) < 2:
            raise ValueError("insufficient observations")

        ordered = list(observations)

        symbol = ordered[0].symbol

        if any(item.symbol != symbol for item in ordered):
            raise ValueError("mixed symbols are not allowed")

        for previous, current in zip(ordered, ordered[1:]):
            if current.timestamp <= previous.timestamp:
                raise ValueError(
                    "observations must be strictly ordered"
                )

        last = ordered[-1]

        if (
            ordered[0].timestamp
            > last.timestamp - self.minimum_history
        ):
            raise ValueError("insufficient time history")

        short_target = last.timestamp - self._short
        long_target = last.timestamp - self._long
        prior_short_start_target = (
            last.timestamp - (self._short * 2)
        )

        short_reference = self._reference(
            ordered,
            target=short_target,
        )

        long_reference = self._reference(
            ordered,
            target=long_target,
        )

        prior_short_start = self._reference(
            ordered,
            target=prior_short_start_target,
        )

        return_short = _return(
            last.price,
            short_reference.price,
        )

        return_long = _return(
            last.price,
            long_reference.price,
        )

        prior_short_return = _return(
            short_reference.price,
            prior_short_start.price,
        )

        acceleration = return_short - prior_short_return

        anchors = self._volatility_anchors(
            ordered,
            last=last,
        )

        returns = [
            _return(current.price, previous.price)
            for previous, current in zip(
                anchors,
                anchors[1:],
            )
        ]

        if not returns:
            raise ValueError(
                "insufficient observations for volatility"
            )

        mean_return = (
            sum(returns, Decimal("0"))
            / Decimal(len(returns))
        )

        variance = (
            sum(
                (
                    (item - mean_return) ** 2
                    for item in returns
                ),
                Decimal("0"),
            )
            / Decimal(len(returns))
        )

        realized_volatility = _sqrt(variance)

        midpoint = (
            last.bid + last.ask
        ) / Decimal("2")

        relative_spread = (
            last.ask - last.bid
        ) / midpoint

        return IntradayFeatures(
            symbol=symbol,
            timestamp=last.timestamp,
            observations=len(anchors),
            last_price=last.price,
            return_short=return_short,
            return_long=return_long,
            acceleration=acceleration,
            realized_volatility=realized_volatility,
            relative_spread=relative_spread,
        )
