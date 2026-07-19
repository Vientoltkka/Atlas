"""Safe parameter reference resolution for execution plan steps."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Mapping


REFERENCE_PATTERN = re.compile(
    r"^steps\.([A-Za-z0-9_-]+)\.output(?:\.([A-Za-z0-9_.-]+))?$"
)
TEMPLATE_REFERENCE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")
BLOCKED_REFERENCE_PARTS = {
    "__class__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "secret",
    "password",
    "token",
    "credentials",
}
MAX_RESOLUTION_DEPTH = 32
MAX_TEMPLATE_LENGTH = 8192
MAX_TEMPLATE_REFERENCES = 32
MAX_INTERPOLATED_RESULT_LENGTH = 65536


class ParameterResolutionErrorCode(str, Enum):
    """Stable error codes for parameter resolution failures."""

    INVALID_REFERENCE_SYNTAX = "INVALID_REFERENCE_SYNTAX"
    REFERENCED_STEP_NOT_FOUND = "REFERENCED_STEP_NOT_FOUND"
    REFERENCED_OUTPUT_MISSING = "REFERENCED_OUTPUT_MISSING"
    REFERENCED_FIELD_NOT_FOUND = "REFERENCED_FIELD_NOT_FOUND"
    INVALID_LIST_INDEX = "INVALID_LIST_INDEX"
    REFERENCE_TO_INCOMPLETE_STEP = "REFERENCE_TO_INCOMPLETE_STEP"
    PARAMETER_RESOLUTION_FAILED = "PARAMETER_RESOLUTION_FAILED"
    INVALID_TEMPLATE_STRUCTURE = "INVALID_TEMPLATE_STRUCTURE"
    INVALID_TEMPLATE_TYPE = "INVALID_TEMPLATE_TYPE"
    INVALID_TEMPLATE_SYNTAX = "INVALID_TEMPLATE_SYNTAX"
    UNRESOLVED_TEMPLATE_REFERENCE = "UNRESOLVED_TEMPLATE_REFERENCE"
    UNSUPPORTED_TEMPLATE_EXPRESSION = "UNSUPPORTED_TEMPLATE_EXPRESSION"
    TEMPLATE_VALUE_NOT_SERIALIZABLE = "TEMPLATE_VALUE_NOT_SERIALIZABLE"
    TEMPLATE_RESOLUTION_FAILED = "TEMPLATE_RESOLUTION_FAILED"


@dataclass(frozen=True, slots=True)
class ParameterResolutionResult:
    """Structured result for argument reference resolution."""

    success: bool
    resolved_arguments: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    used_step_ids: list[str] = field(default_factory=list)
    used_references: list[str] = field(default_factory=list)
    templates_resolved: int = 0
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
        used_references: list[str] = []
        template_counter = [0]

        try:
            resolved = self._resolve_mapping(
                arguments,
                previous_results,
                used_step_ids,
                used_references,
                depth=0,
                seen=set(),
                template_counter=template_counter,
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
                used_references=used_references,
                templates_resolved=template_counter[0],
                error_code=failure.error.code,
                metadata={"resolver": "ParameterResolver"},
            )

        return ParameterResolutionResult(
            success=True,
            resolved_arguments=resolved,
            errors=[],
            unresolved_references=[],
            used_step_ids=used_step_ids,
            used_references=used_references,
            templates_resolved=template_counter[0],
            error_code=None,
            metadata={"resolver": "ParameterResolver"},
        )

    def _resolve_mapping(
        self,
        value: Mapping[str, object],
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        used_references: list[str],
        *,
        depth: int,
        seen: set[int],
        template_counter: list[int],
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
                    used_references,
                    depth=depth + 1,
                    seen=seen,
                    template_counter=template_counter,
                )
        finally:
            seen.remove(id(value))

        return resolved

    def _resolve_value(
        self,
        value: object,
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        used_references: list[str],
        *,
        depth: int,
        seen: set[int],
        template_counter: list[int],
    ) -> object:
        self._guard_depth(depth)

        if isinstance(value, Mapping):
            if "$ref" in value:
                return self._resolve_reference_object(
                    value,
                    previous_results,
                    used_step_ids,
                    used_references,
                )

            if "$template" in value:
                return self._resolve_template_object(
                    value,
                    previous_results,
                    used_step_ids,
                    used_references,
                    template_counter,
                )

            return self._resolve_mapping(
                value,
                previous_results,
                used_step_ids,
                used_references,
                depth=depth,
                seen=seen,
                template_counter=template_counter,
            )

        if isinstance(value, list):
            self._guard_cycle(value, seen)
            try:
                return [
                    self._resolve_value(
                        item,
                        previous_results,
                        used_step_ids,
                        used_references,
                        depth=depth + 1,
                        seen=seen,
                        template_counter=template_counter,
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
                        used_references,
                        depth=depth + 1,
                        seen=seen,
                        template_counter=template_counter,
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
        used_references: list[str],
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
        match = REFERENCE_PATTERN.fullmatch(reference)
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

        used_references.append(reference)

        return deepcopy(value)

    def _resolve_template_object(
        self,
        template_object: Mapping[str, object],
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        used_references: list[str],
        template_counter: list[int],
    ) -> str:
        if tuple(template_object.keys()) != ("$template",):
            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_STRUCTURE.value,
                "$template objects must contain only the '$template' key.",
            )

        template = template_object["$template"]
        if not isinstance(template, str):
            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_TYPE.value,
                "$template value must be a string.",
            )

        if len(template) > MAX_TEMPLATE_LENGTH:
            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value,
                "Template exceeds the maximum supported length.",
            )

        escaped = (
            template
            .replace("{{{{", "\u0000ATLAS_OPEN_BRACE\u0000")
            .replace("}}}}", "\u0000ATLAS_CLOSE_BRACE\u0000")
        )
        matches = list(TEMPLATE_REFERENCE_PATTERN.finditer(escaped))
        if len(matches) > MAX_TEMPLATE_REFERENCES:
            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value,
                "Template exceeds the maximum number of references.",
            )

        self._validate_template_syntax(escaped)

        result_parts: list[str] = []
        cursor = 0
        for match in matches:
            result_parts.append(escaped[cursor:match.start()])
            reference = match.group(1).strip()
            value = self._resolve_template_reference(
                reference,
                previous_results,
                used_step_ids,
                used_references,
            )
            result_parts.append(self._to_template_text(value, reference))
            cursor = match.end()

        result_parts.append(escaped[cursor:])
        interpolated = "".join(result_parts)
        interpolated = (
            interpolated
            .replace("\u0000ATLAS_OPEN_BRACE\u0000", "{{")
            .replace("\u0000ATLAS_CLOSE_BRACE\u0000", "}}")
        )

        if len(interpolated) > MAX_INTERPOLATED_RESULT_LENGTH:
            self._fail(
                ParameterResolutionErrorCode.TEMPLATE_RESOLUTION_FAILED.value,
                "Interpolated template exceeds the maximum supported result length.",
            )

        template_counter[0] += 1
        return interpolated

    def _resolve_template_reference(
        self,
        reference: str,
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        used_references: list[str],
    ) -> object:
        if REFERENCE_PATTERN.fullmatch(reference) is None:
            if any(token in reference for token in ("(", ")", "+", "-", "*", "/", "[", "]", "|", "=", "<", ">")):
                self._fail(
                    ParameterResolutionErrorCode.UNSUPPORTED_TEMPLATE_EXPRESSION.value,
                    f"Unsupported template expression: {self._safe_reference_label(reference)}.",
                    reference,
                )

            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value,
                f"Invalid template reference syntax: {self._safe_reference_label(reference)}.",
                reference,
            )

        try:
            return self._resolve_reference(reference, previous_results, used_step_ids, used_references)
        except _ResolutionFailure as failure:
            self._fail(
                ParameterResolutionErrorCode.UNRESOLVED_TEMPLATE_REFERENCE.value,
                failure.error.message,
                reference,
            )

    def _resolve_reference(
        self,
        reference: str,
        previous_results: Mapping[str, object],
        used_step_ids: list[str],
        used_references: list[str],
    ) -> object:
        match = REFERENCE_PATTERN.fullmatch(reference)
        assert match is not None
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

        used_references.append(reference)
        return value

    def _validate_template_syntax(
        self,
        template: str,
    ) -> None:
        if "{{" in TEMPLATE_REFERENCE_PATTERN.sub("", template):
            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value,
                "Template contains an unclosed reference.",
            )

        if "}}" in TEMPLATE_REFERENCE_PATTERN.sub("", template):
            self._fail(
                ParameterResolutionErrorCode.INVALID_TEMPLATE_SYNTAX.value,
                "Template contains an unopened reference.",
            )

    def _to_template_text(
        self,
        value: object,
        reference: str,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
            try:
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except TypeError:
                self._fail(
                    ParameterResolutionErrorCode.TEMPLATE_VALUE_NOT_SERIALIZABLE.value,
                    "Template value is not JSON serializable.",
                    reference,
                )

        self._fail(
            ParameterResolutionErrorCode.TEMPLATE_VALUE_NOT_SERIALIZABLE.value,
            f"Template value type '{type(value).__name__}' is not supported.",
            reference,
        )

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
            or part in BLOCKED_REFERENCE_PARTS
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
        if depth > MAX_RESOLUTION_DEPTH:
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

    def _safe_reference_label(
        self,
        reference: str,
    ) -> str:
        for sensitive in ("secret", "password", "token", "credentials"):
            reference = re.sub(
                sensitive,
                "[redacted]",
                reference,
                flags=re.IGNORECASE,
            )
        return reference[:120]


class _ResolutionFailure(Exception):
    """Internal control exception for resolution failures."""

    def __init__(
        self,
        error: _ResolutionError,
    ) -> None:
        super().__init__(error.message)
        self.error = error
