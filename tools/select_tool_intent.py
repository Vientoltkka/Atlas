"""Manual validation for deterministic tool selection without execution."""

from __future__ import annotations

import json
import sys

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import ArgumentValidationError
from tools.intent_selector import ToolIntent


def main(argv: list[str] | None = None) -> int:
    """Select a registered tool for one structured intent."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not 1 <= len(args) <= 2:
        print("Usage: python -m tools.select_tool_intent <intent> [json_arguments]")
        return 2

    action = args[0]
    raw_arguments = args[1] if len(args) == 2 else "{}"

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        if "=" not in raw_arguments:
            print(f"Invalid JSON arguments: {error}")
            return 2

        key, value = raw_arguments.split("=", 1)
        if not key:
            print("Argument key cannot be empty.")
            return 2

        arguments = {key: value}

    if not isinstance(arguments, dict):
        print("Arguments must be a JSON object.")
        return 2

    selector = Bootstrap.build_tool_selector()
    validator = Bootstrap.build_argument_validator()
    selection = selector.select(
        ToolIntent(
            action=action,
            arguments=arguments,
        )
    )

    try:
        validation = validator.validate(selection)
    except ArgumentValidationError as error:
        print("Validation error")
        print(f"Intent: {error.intent_action}")
        print(f"Field: {error.field}")
        print(f"Reason: {error.reason}")
        print("Executed: false")
        return 1

    print(f"Intent: {selection.intent.action}")
    print(f"Selected tool: {selection.tool_name}")
    print(
        "Original arguments: "
        + json.dumps(dict(validation.original_arguments), ensure_ascii=False, sort_keys=True)
    )
    print(
        "Validated arguments: "
        + json.dumps(dict(validation.validated_arguments), ensure_ascii=False, sort_keys=True)
    )
    print(f"Valid: {str(validation.valid).lower()}")
    print(f"Executed: {str(validation.executed).lower()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
