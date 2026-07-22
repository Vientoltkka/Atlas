"""Per-tool argument schemas for Atlas tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any

from core.execution_arguments import InvalidExecutionArgumentError, validate_execution_arguments


_MISSING = object()
_SUPPORTED_TYPES = frozenset({bool, int, float, str, list, dict})
_NUMERIC_TYPES = frozenset({int, float})


class ToolSchemaErrorCode(str, Enum):
    """Stable error codes for per-tool argument schema validation."""

    REQUIRED_PARAMETER_MISSING = "REQUIRED_PARAMETER_MISSING"
    UNKNOWN_PARAMETER = "UNKNOWN_PARAMETER"
    INVALID_TYPE = "INVALID_TYPE"
    NONE_NOT_ALLOWED = "NONE_NOT_ALLOWED"
    INVALID_CHOICE = "INVALID_CHOICE"
    BELOW_MINIMUM = "BELOW_MINIMUM"
    ABOVE_MAXIMUM = "ABOVE_MAXIMUM"
    INVALID_SCHEMA = "INVALID_SCHEMA"


@dataclass(frozen=True, slots=True)
class ToolSchemaValidationError:
    """Structured validation error for one tool argument."""

    tool_name: str
    parameter_name: str | None
    error_code: str
    message: str
    received_type: str | None = None
    expected_type: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSchemaValidationResult:
    """Structured result for one per-tool argument validation."""

    is_valid: bool
    normalized_arguments: Mapping[str, object] = field(default_factory=dict)
    errors: tuple[ToolSchemaValidationError, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_arguments",
            MappingProxyType(_copy_mapping(dict(self.normalized_arguments))),
        )
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def valid(self) -> bool:
        """Compatibility alias for callers that prefer a shorter name."""
        return self.is_valid


class ToolSchemaValidationException(ValueError):
    """Raised when a tool call does not match its registered schema."""

    def __init__(
        self,
        result: ToolSchemaValidationResult,
    ) -> None:
        self.result = result
        summary = "; ".join(error.message for error in result.errors)
        super().__init__(summary or "Tool arguments do not match the registered schema.")


@dataclass(frozen=True, slots=True)
class ToolParameterSchema:
    """Schema for one argument accepted by a registered Atlas tool."""

    name: str
    value_type: type
    required: bool = False
    default: object = _MISSING
    allow_none: bool = False
    description: str = ""
    choices: tuple[object, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool parameter name cannot be empty.")
        if self.value_type not in _SUPPORTED_TYPES:
            raise ValueError(f"Unsupported tool parameter type: {self.value_type!r}.")
        if self.minimum is not None and self.value_type not in _NUMERIC_TYPES:
            raise ValueError("minimum is only valid for int and float parameters.")
        if self.maximum is not None and self.value_type not in _NUMERIC_TYPES:
            raise ValueError("maximum is only valid for int and float parameters.")
        if self.minimum is not None and not _is_valid_limit(self.minimum):
            raise ValueError("minimum must be a finite int or float.")
        if self.maximum is not None and not _is_valid_limit(self.maximum):
            raise ValueError("maximum must be a finite int or float.")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot be greater than maximum.")

        if self.choices is not None:
            copied_choices = tuple(_copy_value(choice) for choice in self.choices)
            if not copied_choices:
                raise ValueError("choices cannot be empty when provided.")
            for choice in copied_choices:
                if not _matches_type(choice, self.value_type):
                    raise ValueError("choices must match the parameter value_type.")
            object.__setattr__(self, "choices", copied_choices)

        if self.default is not _MISSING:
            copied_default = _copy_value(self.default)
            error = _validate_parameter_value(
                tool_name="<schema>",
                parameter=self,
                value=copied_default,
            )
            if error is not None:
                raise ValueError(f"default for parameter '{self.name}' is invalid: {error.message}")
            object.__setattr__(self, "default", copied_default)

    @property
    def has_default(self) -> bool:
        """Return whether this parameter declares a static default."""
        return self.default is not _MISSING


@dataclass(frozen=True, slots=True)
class ToolArgumentsSchema:
    """Schema for all arguments accepted by one registered Atlas tool."""

    parameters: tuple[ToolParameterSchema, ...] = ()
    allow_extra_arguments: bool = False

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise ValueError("Tool argument schema cannot contain duplicate parameters.")
        object.__setattr__(self, "parameters", parameters)

    def validate(
        self,
        tool_name: str,
        arguments: Mapping[str, object] | None = None,
    ) -> ToolSchemaValidationResult:
        """Validate and normalize one mapping for this tool."""
        source = dict(arguments or {})
        errors: list[ToolSchemaValidationError] = []
        normalized: dict[str, object] = {}
        parameters_by_name = {parameter.name: parameter for parameter in self.parameters}

        try:
            validate_execution_arguments(source)
        except InvalidExecutionArgumentError as error:
            errors.append(
                ToolSchemaValidationError(
                    tool_name=tool_name,
                    parameter_name=None,
                    error_code=ToolSchemaErrorCode.INVALID_TYPE.value,
                    message=f"tool={tool_name} parameter=<arguments> reason={error}",
                )
            )
            return ToolSchemaValidationResult(False, {}, tuple(errors))

        for name in source:
            if name not in parameters_by_name and not self.allow_extra_arguments:
                errors.append(
                    ToolSchemaValidationError(
                        tool_name=tool_name,
                        parameter_name=name,
                        error_code=ToolSchemaErrorCode.UNKNOWN_PARAMETER.value,
                        message=f"tool={tool_name} parameter={name} reason=unknown parameter",
                    )
                )

        for parameter in self.parameters:
            if parameter.name in source:
                value = source[parameter.name]
            elif parameter.has_default:
                value = _copy_value(parameter.default)
            elif parameter.required:
                errors.append(
                    ToolSchemaValidationError(
                        tool_name=tool_name,
                        parameter_name=parameter.name,
                        error_code=ToolSchemaErrorCode.REQUIRED_PARAMETER_MISSING.value,
                        message=(
                            f"tool={tool_name} parameter={parameter.name} "
                            "reason=required parameter missing"
                        ),
                        expected_type=parameter.value_type.__name__,
                    )
                )
                continue
            else:
                continue

            value_error = _validate_parameter_value(
                tool_name=tool_name,
                parameter=parameter,
                value=value,
            )
            if value_error is not None:
                errors.append(value_error)
                continue

            normalized[parameter.name] = _copy_value(value)

        if self.allow_extra_arguments:
            for name, value in source.items():
                if name not in parameters_by_name:
                    normalized[name] = _copy_value(value)

        return ToolSchemaValidationResult(
            is_valid=not errors,
            normalized_arguments=normalized if not errors else {},
            errors=tuple(errors),
        )


def _validate_parameter_value(
    *,
    tool_name: str,
    parameter: ToolParameterSchema,
    value: object,
) -> ToolSchemaValidationError | None:
    if value is None:
        if parameter.allow_none:
            return None
        return ToolSchemaValidationError(
            tool_name=tool_name,
            parameter_name=parameter.name,
            error_code=ToolSchemaErrorCode.NONE_NOT_ALLOWED.value,
            message=f"tool={tool_name} parameter={parameter.name} reason=None is not allowed",
            received_type="NoneType",
            expected_type=parameter.value_type.__name__,
        )

    if not _matches_type(value, parameter.value_type):
        return ToolSchemaValidationError(
            tool_name=tool_name,
            parameter_name=parameter.name,
            error_code=ToolSchemaErrorCode.INVALID_TYPE.value,
            message=(
                f"tool={tool_name} parameter={parameter.name} "
                f"reason=expected {parameter.value_type.__name__}, got {type(value).__name__}"
            ),
            received_type=type(value).__name__,
            expected_type=parameter.value_type.__name__,
        )

    if parameter.choices is not None and value not in parameter.choices:
        return ToolSchemaValidationError(
            tool_name=tool_name,
            parameter_name=parameter.name,
            error_code=ToolSchemaErrorCode.INVALID_CHOICE.value,
            message=f"tool={tool_name} parameter={parameter.name} reason=invalid choice",
            received_type=type(value).__name__,
            expected_type=parameter.value_type.__name__,
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return ToolSchemaValidationError(
                tool_name=tool_name,
                parameter_name=parameter.name,
                error_code=ToolSchemaErrorCode.INVALID_TYPE.value,
                message=f"tool={tool_name} parameter={parameter.name} reason=non-finite number",
                received_type=type(value).__name__,
                expected_type=parameter.value_type.__name__,
            )
        if parameter.minimum is not None and value < parameter.minimum:
            return ToolSchemaValidationError(
                tool_name=tool_name,
                parameter_name=parameter.name,
                error_code=ToolSchemaErrorCode.BELOW_MINIMUM.value,
                message=f"tool={tool_name} parameter={parameter.name} reason=value below minimum",
                received_type=type(value).__name__,
                expected_type=parameter.value_type.__name__,
            )
        if parameter.maximum is not None and value > parameter.maximum:
            return ToolSchemaValidationError(
                tool_name=tool_name,
                parameter_name=parameter.name,
                error_code=ToolSchemaErrorCode.ABOVE_MAXIMUM.value,
                message=f"tool={tool_name} parameter={parameter.name} reason=value above maximum",
                received_type=type(value).__name__,
                expected_type=parameter.value_type.__name__,
            )

    return None


def _matches_type(
    value: object,
    expected_type: type,
) -> bool:
    if expected_type is bool:
        return type(value) is bool
    if expected_type is int:
        return type(value) is int
    if expected_type is float:
        return type(value) is float
    if expected_type is str:
        return type(value) is str
    if expected_type is list:
        return type(value) is list
    if expected_type is dict:
        return type(value) is dict
    return False


def _is_valid_limit(
    value: object,
) -> bool:
    if type(value) not in {int, float}:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _copy_mapping(
    values: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: _copy_value(value)
        for key, value in values.items()
    }


def _copy_value(
    value: object,
) -> object:
    return deepcopy(value)
