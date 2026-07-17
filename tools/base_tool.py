"""Base class for every Atlas tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tools.tool_context import ToolContext


class BaseTool(ABC):
    """Base interface for every Atlas tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human readable description."""
        ...

    @property
    def requires_confirmation(self) -> bool:
        """Return whether this tool must be explicitly confirmed before execution."""
        return False

    @abstractmethod
    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        """Execute the tool."""
        ...
