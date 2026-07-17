"""Manual runner for one safe structured tool intent."""

from __future__ import annotations

import json
import sys
from typing import Any

from bootstrap.bootstrap import Bootstrap
from tools.intent_selector import ToolIntent
from tools.single_tool_runner import ToolRunResult


def main(argv: list[str] | None = None) -> int:
    """Execute one selected and validated tool intent."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not 1 <= len(args) <= 2:
        print("Usage: python -m tools.run_single_tool <intent> [json_arguments|key=value]")
        return 2

    action = args[0]
    raw_arguments = args[1] if len(args) == 2 else "{}"
    arguments = _parse_arguments(raw_arguments)

    if not isinstance(arguments, dict):
        print("Arguments must be a JSON object.")
        return 2

    runner = Bootstrap.build_single_tool_runner()
    intent = ToolIntent(action=action, arguments=arguments)
    outcome = runner.run(intent)

    _print_outcome(outcome)

    return 0 if outcome.success else 1


def _parse_arguments(raw_arguments: str) -> Any:
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        if "=" not in raw_arguments:
            print(f"Invalid JSON arguments: {error}")
            raise SystemExit(2) from error

        key, value = raw_arguments.split("=", 1)
        if not key:
            print("Argument key cannot be empty.")
            raise SystemExit(2)

        return {key: value}


def _console_text(value: Any) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def _print_outcome(outcome: ToolRunResult) -> None:
    print(f"Intent: {outcome.intent.action}")
    print(f"Tool: {outcome.tool_name or ''}")
    print(f"Success: {str(outcome.success).lower()}")
    print(f"Status: {outcome.status}")
    print(f"Executed: {str(outcome.executed).lower()}")
    print(f"Execution count: {outcome.execution_count}")

    if outcome.success:
        print("Result:")
        print(_console_text(outcome.result))
        return

    if outcome.error_field:
        print(f"Field: {outcome.error_field}")

    print(f"Error: {outcome.error_message or ''}")


if __name__ == "__main__":
    raise SystemExit(main())
