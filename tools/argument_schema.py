"""Structured argument schemas for selected Atlas tool intents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from tools.intent_selector import ToolIntent, ToolSelection


_MISSING = object()


@dataclass(frozen=True, slots=True)
class ArgumentField:
    """Schema for one structured tool argument."""

    name: str
    expected_type: type | tuple[type, ...]
    required: bool = False
    default: Any = _MISSING
    description: str = ""
    allow_none: bool = False
    validator: Callable[[Any], None] | None = None


@dataclass(frozen=True, slots=True)
class ArgumentSchema:
    """Schema for all arguments accepted by one tool intent."""

    intent_action: str
    fields: tuple[ArgumentField, ...] = ()

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]

        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate argument field in schema '{self.intent_action}'.")


@dataclass(frozen=True, slots=True)
class ArgumentValidationResult:
    """Validated arguments for one selected tool intent."""

    intent: ToolIntent
    tool_name: str
    original_arguments: Mapping[str, Any]
    validated_arguments: Mapping[str, Any]
    valid: bool
    executed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_arguments",
            MappingProxyType(dict(self.original_arguments)),
        )
        object.__setattr__(
            self,
            "validated_arguments",
            MappingProxyType(dict(self.validated_arguments)),
        )


class ArgumentSchemaAlreadyRegisteredError(ValueError):
    """Raised when an argument schema is registered twice."""


class ArgumentSchemaNotRegisteredError(RuntimeError):
    """Raised when an intent has no argument schema."""


class ArgumentValidationError(ValueError):
    """Raised when structured tool arguments do not match their schema."""

    def __init__(
        self,
        intent_action: str,
        field: str,
        reason: str,
    ) -> None:
        self.intent_action = intent_action
        self.field = field
        self.reason = reason
        super().__init__(
            f"Intent '{intent_action}', field '{field}': {reason}"
        )


class ArgumentSchemaRegistry:
    """Central source of truth for intent argument schemas."""

    def __init__(self) -> None:
        self._schemas: dict[str, ArgumentSchema] = {}

    def register(
        self,
        schema: ArgumentSchema,
    ) -> None:
        """Register an argument schema for one intent action."""
        if not schema.intent_action:
            raise ValueError("Argument schema intent cannot be empty.")

        if schema.intent_action in self._schemas:
            raise ArgumentSchemaAlreadyRegisteredError(
                f"Argument schema for intent '{schema.intent_action}' is already registered."
            )

        self._schemas[schema.intent_action] = schema

    def exists(
        self,
        intent_action: str,
    ) -> bool:
        """Return whether an intent has an argument schema."""
        return intent_action in self._schemas

    def get(
        self,
        intent_action: str,
    ) -> ArgumentSchema:
        """Return the schema for one intent action."""
        try:
            return self._schemas[intent_action]
        except KeyError as error:
            raise ArgumentSchemaNotRegisteredError(
                f"Argument schema for intent '{intent_action}' is not registered."
            ) from error

    def list(self) -> tuple[str, ...]:
        """Return intent actions with registered schemas."""
        return tuple(sorted(self._schemas.keys()))

    @property
    def schemas(self) -> Mapping[str, ArgumentSchema]:
        """Return a read-only view of schemas."""
        return MappingProxyType(self._schemas)


class ArgumentValidator:
    """Validate selected tool arguments without executing tools."""

    def __init__(
        self,
        schema_registry: ArgumentSchemaRegistry,
    ) -> None:
        self._schema_registry = schema_registry

    def validate(
        self,
        selection: ToolSelection,
    ) -> ArgumentValidationResult:
        """Validate one selected tool intent."""
        schema = self._schema_registry.get(selection.intent.action)
        original_arguments = dict(selection.arguments)
        fields = {field.name: field for field in schema.fields}

        for name in original_arguments:
            if name not in fields:
                raise ArgumentValidationError(
                    selection.intent.action,
                    name,
                    "unexpected argument",
                )

        validated: dict[str, Any] = {}

        for field_schema in schema.fields:
            if field_schema.name in original_arguments:
                value = original_arguments[field_schema.name]
            elif field_schema.default is not _MISSING:
                value = field_schema.default
            elif field_schema.required:
                raise ArgumentValidationError(
                    selection.intent.action,
                    field_schema.name,
                    "required argument is missing",
                )
            else:
                continue

            self._validate_value(selection.intent.action, field_schema, value)
            validated[field_schema.name] = value

        return ArgumentValidationResult(
            intent=selection.intent,
            tool_name=selection.tool_name,
            original_arguments=original_arguments,
            validated_arguments=validated,
            valid=True,
            executed=False,
        )

    def _validate_value(
        self,
        intent_action: str,
        field_schema: ArgumentField,
        value: Any,
    ) -> None:
        if value is None:
            if field_schema.allow_none:
                return

            raise ArgumentValidationError(
                intent_action,
                field_schema.name,
                "None is not allowed",
            )

        if not self._matches_type(value, field_schema.expected_type):
            expected = self._type_name(field_schema.expected_type)
            actual = type(value).__name__
            raise ArgumentValidationError(
                intent_action,
                field_schema.name,
                f"expected {expected}, got {actual}",
            )

        if field_schema.validator is not None:
            try:
                field_schema.validator(value)
            except ValueError as error:
                raise ArgumentValidationError(
                    intent_action,
                    field_schema.name,
                    str(error),
                ) from error

    def _matches_type(
        self,
        value: Any,
        expected_type: type | tuple[type, ...],
    ) -> bool:
        if expected_type is int:
            return type(value) is int

        if isinstance(expected_type, tuple) and int in expected_type and type(value) is bool:
            return False

        return isinstance(value, expected_type)

    def _type_name(
        self,
        expected_type: type | tuple[type, ...],
    ) -> str:
        if isinstance(expected_type, tuple):
            return " or ".join(item.__name__ for item in expected_type)

        return expected_type.__name__


def require_non_empty(value: Any) -> None:
    """Validate that strings and lists are not empty."""
    if len(value) == 0:
        raise ValueError("value cannot be empty")

