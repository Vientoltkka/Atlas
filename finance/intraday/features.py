"""Deterministic rolling intraday feature calculation."""

from __future__ import annotations

from decimal import Decimal, localcontext

from finance.intraday.models import IntradayFeatures, IntradayObservation


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


class IntradayFeatureEngine:
    def __init__(
        self,
        *,
        short_window: int = 3,
        long_window: int = 6,
    ) -> None:
        if short_window < 2:
            raise ValueError("short_window must be >= 2")
        if long_window <= short_window:
            raise ValueError("long_window must be greater than short_window")

        self._short_window = short_window
        self._long_window = long_window

    @property
    def minimum_observations(self) -> int:
        return self._long_window

    def calculate(
        self,
        observations: list[IntradayObservation],
    ) -> IntradayFeatures:
        if len(observations) < self._long_window:
            raise ValueError("insufficient observations")

        window = observations[-self._long_window :]

        symbol = window[0].symbol
        if any(item.symbol != symbol for item in window):
            raise ValueError("mixed symbols are not allowed")

        for previous, current in zip(window, window[1:]):
            if current.timestamp <= previous.timestamp:
                raise ValueError("observations must be strictly ordered")

        last = window[-1]

        short_reference = window[-self._short_window]
        long_reference = window[0]

        return_short = _return(last.price, short_reference.price)
        return_long = _return(last.price, long_reference.price)

        prior_short_end = window[-2]
        prior_short_start_index = len(window) - self._short_window - 1

        if prior_short_start_index < 0:
            raise ValueError("insufficient observations for acceleration")

        prior_short_start = window[prior_short_start_index]
        prior_short_return = _return(
            prior_short_end.price,
            prior_short_start.price,
        )

        acceleration = return_short - prior_short_return

        returns = [
            _return(current.price, previous.price)
            for previous, current in zip(window, window[1:])
        ]

        mean_return = sum(returns, Decimal("0")) / Decimal(len(returns))

        variance = (
            sum(
                ((item - mean_return) ** 2 for item in returns),
                Decimal("0"),
            )
            / Decimal(len(returns))
        )

        realized_volatility = _sqrt(variance)

        midpoint = (last.bid + last.ask) / Decimal("2")
        relative_spread = (last.ask - last.bid) / midpoint

        return IntradayFeatures(
            symbol=symbol,
            timestamp=last.timestamp,
            observations=len(window),
            last_price=last.price,
            return_short=return_short,
            return_long=return_long,
            acceleration=acceleration,
            realized_volatility=realized_volatility,
            relative_spread=relative_spread,
        )
