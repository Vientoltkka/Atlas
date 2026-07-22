"""Shared validation for structured reference paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable


class StructuredReferencePathError(ValueError):
    """Raised when a structured reference path cannot be navigated safely."""

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def normalize_reference_path(
    raw_path: Iterable[str | int],
    *,
    label: str,
) -> tuple[str | int, ...]:
    """Return a canonical safe path for structured references."""
    if isinstance(raw_path, (str, bytes)):
        raise ValueError(f"{label} path must be an iterable of segments.")

    try:
        normalized_path = tuple(raw_path)
    except TypeError as error:
        raise ValueError(f"{label} path must be iterable.") from error

    for segment in normalized_path:
        if isinstance(segment, bool):
            raise ValueError(f"{label} path bool segments are not allowed.")
        if isinstance(segment, int):
            if segment < 0:
                raise ValueError(f"{label} path indexes cannot be negative.")
            continue
        if isinstance(segment, str):
            if not segment:
                raise ValueError(f"{label} path string segments cannot be empty.")
            if segment.startswith("_"):
                raise ValueError(f"{label} path private segments are not allowed.")
            continue
        raise ValueError(f"{label} path segments must be str or int.")

    return normalized_path


def navigate_structured_path(
    value: object,
    path: tuple[str | int, ...],
    *,
    owner_label: str,
) -> object:
    """Navigate a structural value through mappings and lists only."""
    for part in path:
        if isinstance(value, Mapping):
            if not isinstance(part, str):
                raise StructuredReferencePathError(
                    "REFERENCE_TYPE_ERROR",
                    f"{owner_label} expected a string key at segment '{part}'.",
                )
            if part in value:
                value = value[part]
                continue
            raise StructuredReferencePathError(
                "REFERENCED_FIELD_NOT_FOUND",
                f"Referenced field '{part}' was not found in {owner_label}.",
            )

        if isinstance(value, (list, tuple)):
            if isinstance(part, bool) or not isinstance(part, int):
                raise StructuredReferencePathError(
                    "INVALID_LIST_INDEX",
                    f"Invalid list index '{part}' in {owner_label}.",
                )
            if part < 0 or part >= len(value):
                raise StructuredReferencePathError(
                    "INVALID_LIST_INDEX",
                    f"List index '{part}' is out of range for {owner_label}.",
                )
            value = value[part]
            continue

        raise StructuredReferencePathError(
            "REFERENCE_TYPE_ERROR",
            (
                f"Cannot navigate segment '{part}' in {owner_label} "
                f"through value type '{type(value).__name__}'."
            ),
        )

    return value
