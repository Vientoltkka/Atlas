"""Structured references to outputs produced by earlier execution steps."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class StepOutputReference:
    """Reference a previous step output, optionally selecting a safe path."""

    step_id: str
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("StepOutputReference step_id must be a non-empty string.")

        raw_path = self.path
        if isinstance(raw_path, (str, bytes)):
            raise ValueError("StepOutputReference path must be an iterable of segments.")

        try:
            normalized_path = tuple(raw_path)
        except TypeError as error:
            raise ValueError("StepOutputReference path must be iterable.") from error

        for segment in normalized_path:
            if isinstance(segment, bool):
                raise ValueError("StepOutputReference path bool segments are not allowed.")
            if isinstance(segment, int):
                if segment < 0:
                    raise ValueError("StepOutputReference path indexes cannot be negative.")
                continue
            if isinstance(segment, str):
                if not segment:
                    raise ValueError("StepOutputReference path string segments cannot be empty.")
                if segment.startswith("_"):
                    raise ValueError("StepOutputReference path private segments are not allowed.")
                continue
            raise ValueError("StepOutputReference path segments must be str or int.")

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
