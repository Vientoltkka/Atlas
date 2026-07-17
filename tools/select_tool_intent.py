"""Manual validation for deterministic tool selection without execution."""

from __future__ import annotations

import json
import sys

from bootstrap.bootstrap import Bootstrap
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
    selection = selector.select(
        ToolIntent(
            action=action,
            arguments=arguments,
        )
    )

    print(f"Intent: {selection.intent.action}")
    print(f"Selected tool: {selection.tool_name}")
    print(
        "Arguments: "
        + json.dumps(dict(selection.arguments), ensure_ascii=False, sort_keys=True)
    )
    print(f"Executed: {str(selection.executed).lower()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
