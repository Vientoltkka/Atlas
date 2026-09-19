"""Intraday signal orchestration without execution capability."""

from __future__ import annotations

from finance.intraday.buffer import IntradayObservationBuffer
from finance.intraday.features import IntradayFeatureEngine
from finance.intraday.models import IntradayObservation, IntradaySignal
from finance.intraday.signal_engine import IntradaySignalEngine


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
