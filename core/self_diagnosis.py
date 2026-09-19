from __future__ import annotations

from dataclasses import dataclass

from core.model_latency import ModelLatencyTracker


@dataclass(frozen=True, slots=True)
class SelfDiagnosisFinding:
    finding_id: str
    title: str
    evidence: tuple[str, ...]
    objective: str
    risk: str


class SelfDiagnosisService:
    """Read-only diagnosis of improvement opportunities from runtime evidence."""

    def __init__(
        self,
        *,
        latency_tracker: ModelLatencyTracker | None = None,
        minimum_observations: int = 2,
        slow_latency_threshold_seconds: float = 5.0,
    ) -> None:
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be >= 1.")
        if slow_latency_threshold_seconds < 0:
            raise ValueError("slow_latency_threshold_seconds must be >= 0.")

        self._latency_tracker = latency_tracker
        self._minimum_observations = minimum_observations
        self._slow_latency_threshold_seconds = float(
            slow_latency_threshold_seconds
        )

    def diagnose(self) -> tuple[SelfDiagnosisFinding, ...]:
        if self._latency_tracker is None:
            return ()

        candidates = tuple(
            sample
            for sample in self._latency_tracker.snapshot()
            if (
                sample.observation_count >= self._minimum_observations
                and sample.latency_seconds
                >= self._slow_latency_threshold_seconds
            )
        )

        if not candidates:
            return ()

        slowest = max(candidates, key=lambda item: item.latency_seconds)

        return (
            SelfDiagnosisFinding(
                finding_id="runtime.model-latency",
                title="Latencia de inferencia observada",
                evidence=(
                    f"modelo={slowest.model_name}",
                    f"proveedor={slowest.provider_id or 'desconocido'}",
                    f"latencia_observada={slowest.latency_seconds:.3f}s",
                    f"observaciones={slowest.observation_count}",
                ),
                objective=(
                    "Reducir la latencia de respuesta utilizando "
                    "evidencia real de ejecuci?n."
                ),
                risk=(
                    "Diagn?stico de solo lectura; "
                    "no autoriza ni aplica cambios."
                ),
            ),
        )
