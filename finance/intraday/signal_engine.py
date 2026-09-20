"""Deterministic V1 intraday signal policy.

Thresholds are initial research parameters, not profitability claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from finance.intraday.models import (
    IntradayFeatures,
    IntradaySignal,
    IntradaySignalAction,
)


@dataclass(frozen=True)
class IntradaySignalPolicy:
    minimum_short_return: Decimal = Decimal("0.0010")
    minimum_long_return: Decimal = Decimal("0.0020")
    minimum_acceleration: Decimal = Decimal("0")
    maximum_volatility: Decimal = Decimal("0.0100")
    maximum_relative_spread: Decimal = Decimal("0.0030")


class IntradaySignalEngine:
    def __init__(
        self,
        policy: IntradaySignalPolicy | None = None,
    ) -> None:
        self._policy = policy or IntradaySignalPolicy()

    @property
    def policy(self) -> IntradaySignalPolicy:
        return self._policy

    def evaluate(self, features: IntradayFeatures) -> IntradaySignal:
        reasons: list[str] = []

        if features.return_short < self._policy.minimum_short_return:
            reasons.append("SHORT_MOMENTUM_TOO_LOW")

        if features.return_long < self._policy.minimum_long_return:
            reasons.append("LONG_MOMENTUM_TOO_LOW")

        if features.acceleration <= self._policy.minimum_acceleration:
            reasons.append("NO_POSITIVE_ACCELERATION")

        if features.realized_volatility > self._policy.maximum_volatility:
            reasons.append("VOLATILITY_TOO_HIGH")

        if features.relative_spread > self._policy.maximum_relative_spread:
            reasons.append("SPREAD_TOO_WIDE")

        action = (
            IntradaySignalAction.CANDIDATE
            if not reasons
            else IntradaySignalAction.NO_ACTION
        )

        return IntradaySignal(
            symbol=features.symbol,
            timestamp=features.timestamp,
            action=action,
            reasons=tuple(reasons) if reasons else ("POLICY_MATCH",),
            features=features,
            # The current policy only accepts positive momentum, so its
            # deterministic directional decision is LONG.
            direction="LONG" if action is IntradaySignalAction.CANDIDATE else None,
        )
