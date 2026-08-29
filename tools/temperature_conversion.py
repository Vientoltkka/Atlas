"""Deterministic Celsius-to-Fahrenheit conversion tool."""

from __future__ import annotations

from tools.base_tool import BaseTool
from tools.tool_context import ToolContext


class TemperatureConversionTool(BaseTool):
    """Convert Celsius values to Fahrenheit without external effects."""

    @property
    def name(self) -> str:
        return "temperature_conversion"

    @property
    def description(self) -> str:
        return "Convert Celsius to Fahrenheit deterministically."

    def execute(self, context: ToolContext) -> float:
        value = context.parameters.get("celsius")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Missing numeric parameter 'celsius'.")
        return float(value) * 9 / 5 + 32