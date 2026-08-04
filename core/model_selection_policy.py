"""Explicit runtime preferences for deterministic model selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.model_manager import ModelSelectionRequest


@dataclass(frozen=True, slots=True)
class ModelSelectionPolicy:
    """Runtime preferences that are transported to the model selector."""

    preferred_provider: str | None = None
    prefer_local: bool | None = None
    max_cost: float | None = None
    max_latency: float | None = None
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        if self.preferred_provider is not None:
            if (
                not isinstance(self.preferred_provider, str)
                or not self.preferred_provider.strip()
            ):
                raise ValueError(
                    "preferred_provider must be a non-empty string when provided."
                )
            object.__setattr__(
                self,
                "preferred_provider",
                self.preferred_provider.strip(),
            )
        if self.prefer_local is not None and type(self.prefer_local) is not bool:
            raise TypeError("prefer_local must be a bool or None.")
        if type(self.allow_fallback) is not bool:
            raise TypeError("allow_fallback must be a bool.")
        for name in ("max_cost", "max_latency"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative when provided.")

    def create_request(
        self,
        *,
        task: str,
        preferred_model_id: str | None = None,
    ) -> ModelSelectionRequest:
        """Create one isolated request without making a selection decision."""
        return ModelSelectionRequest(
            task=task,
            prefer_local=self.prefer_local,
            maximum_relative_cost=self.max_cost,
            maximum_relative_latency=self.max_latency,
            preferred_model_id=preferred_model_id,
            preferred_provider_id=self.preferred_provider,
            allow_fallback=self.allow_fallback,
        )
