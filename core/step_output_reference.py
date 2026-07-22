"""Structured references to outputs produced by earlier execution steps."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from core.structured_reference_path import normalize_reference_path


@dataclass(frozen=True, slots=True)
class StepOutputReference:
    """Reference a previous step output, optionally selecting a safe path."""

    step_id: str
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("StepOutputReference step_id must be a non-empty string.")

        normalized_path = normalize_reference_path(
            self.path,
            label="StepOutputReference",
        )
        object.__setattr__(self, "step_id", self.step_id.strip())
        object.__setattr__(self, "path", normalized_path)

    @classmethod
    def from_path(
        cls,
        step_id: str,
        path: Iterable[str | int] = (),
    ) -> "StepOutputReference":
        """Build a reference from any iterable path."""
        return cls(step_id=step_id, path=tuple(path))


def copy_step_output_reference(
    reference: StepOutputReference,
) -> StepOutputReference:
    """Return a defensive copy of a reference."""
    return StepOutputReference(reference.step_id, deepcopy(reference.path))
