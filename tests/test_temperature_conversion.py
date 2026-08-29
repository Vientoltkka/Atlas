from __future__ import annotations

from bootstrap.bootstrap import Bootstrap
from tools.temperature_conversion import TemperatureConversionTool
from tools.tool_context import ToolContext


def test_converts_celsius_to_fahrenheit_deterministically() -> None:
    tool = TemperatureConversionTool()
    assert tool.execute(ToolContext({"celsius": 37})) == 98.6
    assert tool.execute(ToolContext({"celsius": 0})) == 32.0


def test_temperature_conversion_tool_is_registered() -> None:
    assert isinstance(Bootstrap.build_tool_registry().get("temperature_conversion"), TemperatureConversionTool)