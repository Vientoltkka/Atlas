"""Central registry for Atlas tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from tools.base_tool import BaseTool
from tools.tool_schema import ToolArgumentsSchema


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Public metadata for a registered Atlas tool."""

    name: str
    description: str
    tool: BaseTool
    requires_confirmation: bool = False
    argument_names: tuple[str, ...] = ()
    required_arguments: tuple[str, ...] = ()
    optional_arguments: tuple[str, ...] = ()
    dangerous: bool = False
    output_description: str | None = None
    arguments_schema: ToolArgumentsSchema | None = None


class ToolAlreadyRegisteredError(ValueError):
    """Raised when a tool identifier is registered twice."""


class ToolNotRegisteredError(RuntimeError):
    """Raised when a tool identifier is not present in the registry."""


class ToolRegistry:
    """Stores every available Atlas tool in one central collection."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._argument_schemas: dict[str, ToolArgumentsSchema] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        arguments_schema: ToolArgumentsSchema | None = None,
    ) -> None:
        """Register a tool."""
        name = tool.name

        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool '{name}' is already registered."
            )

        schema = arguments_schema
        if schema is None:
            candidate = getattr(tool, "arguments_schema", None)
            if isinstance(candidate, ToolArgumentsSchema):
                schema = candidate

        self._tools[name] = tool
        if schema is not None:
            self._argument_schemas[name] = schema

    def get(self, name: str) -> BaseTool:
        """Return a registered tool by name."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotRegisteredError(
                f"Tool '{name}' is not registered."
            ) from error

    def descriptor(self, name: str) -> ToolDescriptor:
        """Return public metadata for a registered tool."""
        tool = self.get(name)

        return ToolDescriptor(
            name=tool.name,
            description=tool.description,
            tool=tool,
            requires_confirmation=tool.requires_confirmation,
            dangerous=tool.requires_confirmation,
            arguments_schema=self.arguments_schema(name),
        )

    def arguments_schema(
        self,
        name: str,
    ) -> ToolArgumentsSchema | None:
        """Return the registered argument schema for one tool, if any."""
        self.get(name)
        return self._argument_schemas.get(name)

    def exists(self, name: str) -> bool:
        """Check if a tool exists."""

        return name in self._tools

    def list(self) -> tuple[str, ...]:
        """Return registered tool names."""

        return tuple(sorted(self._tools.keys()))

    def descriptors(self) -> tuple[ToolDescriptor, ...]:
        """Return public metadata for all registered tools."""

        return tuple(
            self.descriptor(name)
            for name in self.list()
        )

    @property
    def tools(self) -> Mapping[str, BaseTool]:
        """Return a read-only view of the registered tool instances."""

        return MappingProxyType(self._tools)
