"""Manual runner for one safe structured tool intent."""

from __future__ import annotations

import json
import sys
from typing import Any

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import ArgumentValidationError
from tools.intent_selector import ToolIntent


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

    try:
        result = runner.run(intent)
    except ArgumentValidationError as error:
        print("Validation error")
        print(f"Intent: {error.intent_action}")
        print(f"Field: {error.field}")
        print(f"Reason: {error.reason}")
        print("Executed: false")
        return 1

    request = runner.last_request
    if request is None:
        raise RuntimeError("Single tool runner finished without a validated request.")

    print(f"Intent: {request.intent.action}")
    print(f"Selected tool: {request.tool_name}")
    print(f"Validated: {str(request.validated).lower()}")
    print("Executed: true")
    print(f"Execution count: {runner.execution_count}")
    print("Result:")
    print(_console_text(result))

    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
