"""Shared validation for structured reference paths."""

from __future__ import annotations

from typing import Iterable


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
