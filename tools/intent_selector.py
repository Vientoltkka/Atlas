"""Deterministic selection of registered tools from structured intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from tools.registry import ToolDescriptor, ToolRegistry, ToolNotRegisteredError


@dataclass(frozen=True, slots=True)
class ToolIntent:
    """Structured request for selecting a tool without executing it."""

    action: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    target: str | None = None

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("Tool intent action cannot be empty.")

        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """Result of resolving one structured intent to one registered tool."""

    intent: ToolIntent
    tool_name: str
    descriptor: ToolDescriptor
    arguments: Mapping[str, Any]
    executed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


class ToolIntentAlreadyRegisteredError(ValueError):
    """Raised when an intent action is mapped twice."""


class ToolIntentNotSupportedError(RuntimeError):
    """Raised when an intent action has no registered mapping."""


class ToolIntentRegistry:
    """Central source of truth for intent-to-tool mappings."""

    def __init__(self) -> None:
        self._mappings: dict[str, str] = {}

    def register(
        self,
        action: str,
        tool_name: str,
    ) -> None:
        """Register one stable intent action to one tool identifier."""
        if not action:
            raise ValueError("Tool intent action cannot be empty.")

        if not tool_name:
            raise ValueError("Tool name cannot be empty.")

        if action in self._mappings:
            raise ToolIntentAlreadyRegisteredError(
                f"Tool intent '{action}' is already registered."
            )

        self._mappings[action] = tool_name

    def supports(
        self,
        action: str,
    ) -> bool:
        """Return whether an action has an explicit tool mapping."""
        return action in self._mappings

    def resolve(
        self,
        action: str,
    ) -> str:
        """Return the mapped tool identifier for an action."""
        try:
            return self._mappings[action]
        except KeyError as error:
            raise ToolIntentNotSupportedError(
                f"Tool intent '{action}' is not supported."
            ) from error

    def list(self) -> tuple[str, ...]:
        """Return supported intent actions."""
        return tuple(sorted(self._mappings.keys()))

    @property
    def mappings(self) -> Mapping[str, str]:
        """Return a read-only view of intent-to-tool mappings."""
        return MappingProxyType(self._mappings)


class ToolSelector:
    """Select registered tools from structured intents without executing them."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        intent_registry: ToolIntentRegistry,
    ) -> None:
        self._tool_registry = tool_registry
        self._intent_registry = intent_registry

    def supports(
        self,
        action: str,
    ) -> bool:
        """Return whether an intent action can be selected."""
        return self._intent_registry.supports(action)

    def supported_intents(self) -> tuple[str, ...]:
        """Return all supported intent actions."""
        return self._intent_registry.list()

    def select(
        self,
        intent: ToolIntent,
    ) -> ToolSelection:
        """Resolve an intent to a registered tool descriptor without execution."""
        tool_name = self._intent_registry.resolve(intent.action)

        try:
            descriptor = self._tool_registry.descriptor(tool_name)
        except ToolNotRegisteredError as error:
            raise ToolNotRegisteredError(
                f"Tool intent '{intent.action}' maps to missing tool '{tool_name}'."
            ) from error

        return ToolSelection(
            intent=intent,
            tool_name=tool_name,
            descriptor=descriptor,
            arguments=intent.arguments,
            executed=False,
        )

