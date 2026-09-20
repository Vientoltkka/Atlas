"""Intraday signal orchestration without execution capability."""

from __future__ import annotations

from finance.intraday.buffer import IntradayObservationBuffer
from finance.intraday.features import IntradayFeatureEngine
from finance.intraday.models import IntradayObservation, IntradaySignal
from finance.intraday.signal_engine import IntradaySignalEngine
from finance.intraday.time_features import TimeBasedIntradayFeatureEngine


class IntradaySignalService:
    def __init__(
        self,
        *,
        buffer: IntradayObservationBuffer | None = None,
        feature_engine: IntradayFeatureEngine | None = None,
        signal_engine: IntradaySignalEngine | None = None,
    ) -> None:
        self._buffer = buffer or IntradayObservationBuffer()
        self._feature_engine = feature_engine or IntradayFeatureEngine()
        self._signal_engine = signal_engine or IntradaySignalEngine()

    def ingest(
        self,
        observation: IntradayObservation,
    ) -> IntradaySignal | None:
        if not self._buffer.add(observation):
            return None

        observations = self._buffer.observations(observation.symbol)

        if len(observations) < self._feature_engine.minimum_observations:
            return None

        features = self._feature_engine.calculate(observations)
        return self._signal_engine.evaluate(features)


class TimeBasedIntradaySignalService:
    """V2 signal service with the V1 signal policy and time-based features."""

    def __init__(
        self,
        *,
        feature_engine: TimeBasedIntradayFeatureEngine | None = None,
        signal_engine: IntradaySignalEngine | None = None,
    ) -> None:
        self._feature_engine = feature_engine or TimeBasedIntradayFeatureEngine()
        self._signal_engine = signal_engine or IntradaySignalEngine()
        self._observations: dict[str, list[IntradayObservation]] = {}

    @property
    def configuration(self) -> dict[str, object]:
        policy = self._signal_engine.policy
        return {
            "short_minutes": self._feature_engine.short_minutes,
            "long_minutes": self._feature_engine.long_minutes,
            "maximum_reference_lateness_seconds": (
                self._feature_engine.maximum_reference_lateness_seconds
            ),
            "volatility_step_seconds": (
                self._feature_engine.volatility_step_seconds
            ),
            "minimum_short_return": policy.minimum_short_return,
            "minimum_long_return": policy.minimum_long_return,
            "minimum_acceleration": policy.minimum_acceleration,
            "maximum_volatility": policy.maximum_volatility,
            "maximum_relative_spread": policy.maximum_relative_spread,
        }

    def ingest(
        self,
        observation: IntradayObservation,
    ) -> IntradaySignal | None:
        observations = self._observations.setdefault(observation.symbol, [])
        if observations and observation.timestamp <= observations[-1].timestamp:
            return None

        observations.append(observation)
        try:
            features = self._feature_engine.calculate(observations)
        except ValueError:
            return None
        return self._signal_engine.evaluate(features)
