"""Safe parameter reference resolution for execution plan steps."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping


class ParameterResolutionErrorCode(str, Enum):
    """Stable error codes for parameter resolution failures."""

    INVALID_REFERENCE_SYNTAX = "INVALID_REFERENCE_SYNTAX"
    REFERENCED_STEP_NOT_FOUND = "REFERENCED_STEP_NOT_FOUND"
    REFERENCED_OUTPUT_MISSING = "REFERENCED_OUTPUT_MISSING"
    REFERENCED_FIELD_NOT_FOUND = "REFERENCED_FIELD_NOT_FOUND"
    INVALID_LIST_INDEX = "INVALID_LIST_INDEX"
    REFERENCE_TO_INCOMPLETE_STEP = "REFERENCE_TO_INCOMPLETE_STEP"
    PARAMETER_RESOLUTION_FAILED = "PARAMETER_RESOLUTION_FAILED"


@dataclass(frozen=True, slots=True)
class ParameterResolutionResult:
    """Structured result for argument reference resolution."""

    success: bool
    resolved_arguments: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    used_step_ids: list[str] = field(default_factory=list)
    error_code: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ResolutionError:
    code: str
    message: str
    reference: str | None = None


class ParameterResolver:
    """Resolve explicit step-output references without executing code."""

    _REFERENCE_PATTERN = re.compile(
        r"^steps\.([A-Za-z0-9_-]+)\.output(?:\.([A-Za-z0-9_.-]+))?$"
    )
    _BLOCKED_PATH_PARTS = {"__class__", "__dict__", "__globals__"}
    _MAX_DEPTH = 32

    def resolve(
        self,
        arguments: Mapping[str, object],
        previous_results: Mapping[str, object],
    ) -> ParameterResolutionResult:
        """Return a new argument mapping with explicit references resolved."""
        used_step_ids: list[str] = []

        try:
            resolved = self._resolve_mapping(
                arguments,
                previous_results,
                used_step_ids,
                depth=0,
                seen=set(),
            )
        except _ResolutionFailure as failure:
            return ParameterResolutionResult(
                success=False,
                resolved_arguments={},
                errors=[failure.error.message],
                unresolved_references=(
                    [failure.error.reference]
                    if failure.error.reference is not None
                    else []
                ),
                used_step_ids=used_step_ids,
                error_code=failure.error.code,
                metadata={"resolver": "ParameterResolver"},
            )

        return ParameterResolutionResult(
            success=True,
            resolved_arguments=resolved,
            errors=[],
            unresolved_references=[],
            used_step_ids=used_step_ids,
            error_code=None,
            metadata={"resolver": "ParameterResolver"},
        )

    def _resolve_mapping(
        self,
        value: Mapping[str, object],
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        *,
        depth: int,
        seen: set[int],
    ) -> dict[str, object]:
        self._guard_depth(depth)
        self._guard_cycle(value, seen)

        resolved: dict[str, object] = {}
        try:
            for key, item in value.items():
                resolved[key] = self._resolve_value(
                    item,
                    previous_results,
                    used_step_ids,
                    depth=depth + 1,
                    seen=seen,
                )
        finally:
            seen.remove(id(value))

        return resolved

    def _resolve_value(
        self,
        value: object,
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        *,
        depth: int,
        seen: set[int],
    ) -> object:
        self._guard_depth(depth)

        if isinstance(value, Mapping):
            if "$ref" in value:
                return self._resolve_reference_object(
                    value,
                    previous_results,
                    used_step_ids,
                )

            return self._resolve_mapping(
                value,
                previous_results,
                used_step_ids,
                depth=depth,
                seen=seen,
            )

        if isinstance(value, list):
            self._guard_cycle(value, seen)
            try:
                return [
                    self._resolve_value(
                        item,
                        previous_results,
                        used_step_ids,
                        depth=depth + 1,
                        seen=seen,
                    )
                    for item in value
                ]
            finally:
                seen.remove(id(value))

        if isinstance(value, tuple):
            self._guard_cycle(value, seen)
            try:
                return tuple(
                    self._resolve_value(
                        item,
                        previous_results,
                        used_step_ids,
                        depth=depth + 1,
                        seen=seen,
                    )
                    for item in value
                )
            finally:
                seen.remove(id(value))

        return deepcopy(value)

    def _resolve_reference_object(
        self,
        reference_object: Mapping[str, object],
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
    ) -> object:
        if tuple(reference_object.keys()) != ("$ref",):
            self._fail(
                ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value,
                "$ref objects must contain only the '$ref' key.",
            )

        raw_reference = reference_object["$ref"]
        if not isinstance(raw_reference, str) or not raw_reference.strip():
            self._fail(
                ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value,
                "$ref value must be a non-empty string.",
            )

        reference = raw_reference.strip()
        match = self._REFERENCE_PATTERN.fullmatch(reference)
        if match is None:
            self._fail(
                ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value,
                f"Invalid reference syntax: {reference}.",
                reference,
            )

        step_id = match.group(1)
        path = match.group(2)

        if step_id not in previous_results:
            self._fail(
                ParameterResolutionErrorCode.REFERENCED_STEP_NOT_FOUND.value,
                f"Referenced step '{step_id}' was not found in previous results.",
                reference,
            )

        output = self._extract_output(previous_results[step_id], reference)
        value = output

        if path is not None:
            value = self._resolve_path(value, step_id, path, reference)

        if step_id not in used_step_ids:
            used_step_ids.append(step_id)

        return deepcopy(value)

    def _extract_output(
        self,
        result: object,
        reference: str,
    ) -> object:
        if self._looks_like_step_execution_result(result):
            step_id = str(getattr(result, "step_id"))
            if not getattr(result, "success") or getattr(result, "status") != "completed":
                self._fail(
                    ParameterResolutionErrorCode.REFERENCE_TO_INCOMPLETE_STEP.value,
                    f"Referenced step '{step_id}' is not completed.",
                    reference,
                )

            output = getattr(result, "output")
            if output is None:
                self._fail(
                    ParameterResolutionErrorCode.REFERENCED_OUTPUT_MISSING.value,
                    f"Referenced step '{step_id}' has no output.",
                    reference,
                )

            return output

        return result

    def _resolve_path(
        self,
        value: object,
        step_id: str,
        path: str,
        reference: str,
    ) -> object:
        for part in path.split("."):
            self._validate_path_part(part, reference)

            if isinstance(value, Mapping):
                if part in value:
                    value = value[part]
                    continue

                self._fail(
                    ParameterResolutionErrorCode.REFERENCED_FIELD_NOT_FOUND.value,
                    f"Referenced field '{part}' was not found in step '{step_id}'.",
                    reference,
                )

            if isinstance(value, list):
                if not part.isdigit():
                    self._fail(
                        ParameterResolutionErrorCode.INVALID_LIST_INDEX.value,
                        f"Invalid list index '{part}' in reference to step '{step_id}'.",
                        reference,
                    )

                index = int(part)
                if index >= len(value):
                    self._fail(
                        ParameterResolutionErrorCode.INVALID_LIST_INDEX.value,
                        f"List index '{part}' is out of range in reference to step '{step_id}'.",
                        reference,
                    )

                value = value[index]
                continue

            if isinstance(value, tuple):
                if not part.isdigit():
                    self._fail(
                        ParameterResolutionErrorCode.INVALID_LIST_INDEX.value,
                        f"Invalid tuple index '{part}' in reference to step '{step_id}'.",
                        reference,
                    )

                index = int(part)
                if index >= len(value):
                    self._fail(
                        ParameterResolutionErrorCode.INVALID_LIST_INDEX.value,
                        f"Tuple index '{part}' is out of range in reference to step '{step_id}'.",
                        reference,
                    )

                value = value[index]
                continue

            if hasattr(value, part):
                value = getattr(value, part)
                continue

            self._fail(
                ParameterResolutionErrorCode.REFERENCED_FIELD_NOT_FOUND.value,
                f"Referenced field '{part}' was not found in step '{step_id}'.",
                reference,
            )

        return value

    def _validate_path_part(
        self,
        part: str,
        reference: str,
    ) -> None:
        if (
            not part
            or part.startswith("_")
            or part in self._BLOCKED_PATH_PARTS
            or not re.fullmatch(r"[A-Za-z0-9_-]+", part)
        ):
            self._fail(
                ParameterResolutionErrorCode.INVALID_REFERENCE_SYNTAX.value,
                f"Unsafe reference path segment: {part}.",
                reference,
            )

    def _guard_depth(
        self,
        depth: int,
    ) -> None:
        if depth > self._MAX_DEPTH:
            self._fail(
                ParameterResolutionErrorCode.PARAMETER_RESOLUTION_FAILED.value,
                "Parameter resolution exceeded the maximum supported depth.",
            )

    def _guard_cycle(
        self,
        value: object,
        seen: set[int],
    ) -> None:
        value_id = id(value)
        if value_id in seen:
            self._fail(
                ParameterResolutionErrorCode.PARAMETER_RESOLUTION_FAILED.value,
                "Circular parameter structure detected.",
            )

        seen.add(value_id)

    def _fail(
        self,
        code: str,
        message: str,
        reference: str | None = None,
    ) -> None:
        raise _ResolutionFailure(
            _ResolutionError(
                code=code,
                message=message,
                reference=reference,
            )
        )

    def _looks_like_step_execution_result(
        self,
        result: object,
    ) -> bool:
        return (
            hasattr(result, "step_id")
            and hasattr(result, "status")
            and hasattr(result, "success")
            and hasattr(result, "output")
        )


class _ResolutionFailure(Exception):
    """Internal control exception for resolution failures."""

    def __init__(
        self,
        error: _ResolutionError,
    ) -> None:
        super().__init__(error.message)
        self.error = error
