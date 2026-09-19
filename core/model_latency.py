from __future__ import annotations

from dataclasses import dataclass
import math
from threading import RLock


@dataclass(frozen=True, slots=True)
class ModelLatencySample:
    provider_id: str | None
    model_name: str
    latency_seconds: float
    observation_count: int


class ModelLatencyTracker:
    """Keep bounded in-memory latency estimates per provider/model pair."""

    def __init__(self, *, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self._alpha = float(alpha)
        self._estimates: dict[tuple[str | None, str], float] = {}
        self._counts: dict[tuple[str | None, str], int] = {}
        self._lock = RLock()

    def record(
        self,
        model_name: str,
        provider_id: str | None,
        latency_seconds: float,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if (
            isinstance(latency_seconds, bool)
            or not isinstance(latency_seconds, (int, float))
            or not math.isfinite(latency_seconds)
            or latency_seconds < 0
        ):
            raise ValueError("latency_seconds must be finite and non-negative.")

        key = (provider_id, model_name.strip())
        value = float(latency_seconds)

        with self._lock:
            previous = self._estimates.get(key)
            self._estimates[key] = (
                value
                if previous is None
                else self._alpha * value + (1.0 - self._alpha) * previous
            )
            self._counts[key] = self._counts.get(key, 0) + 1

    def estimate(
        self,
        model_name: str,
        provider_id: str | None,
    ) -> float | None:
        with self._lock:
            return self._estimates.get((provider_id, model_name.strip()))

    def snapshot(self) -> tuple[ModelLatencySample, ...]:
        """Return an immutable point-in-time view of observed model latencies."""
        with self._lock:
            items = tuple(
                (
                    provider_id,
                    model_name,
                    latency_seconds,
                    self._counts.get((provider_id, model_name), 0),
                )
                for (provider_id, model_name), latency_seconds
                in self._estimates.items()
            )

        return tuple(
            ModelLatencySample(
                provider_id=provider_id,
                model_name=model_name,
                latency_seconds=latency_seconds,
                observation_count=observation_count,
            )
            for (
                provider_id,
                model_name,
                latency_seconds,
                observation_count,
            ) in items
        )
