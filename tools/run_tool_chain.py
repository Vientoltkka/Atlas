"""Manual runner for deterministic linear tool chains."""

from __future__ import annotations

import json
import sys
from typing import Any

from bootstrap.bootstrap import Bootstrap
from tools.tool_chain_runner import ToolChainResult, ToolChainStep


def main(argv: list[str] | None = None) -> int:
    """Execute one deterministic chain from JSON."""
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) != 1:
        print("Usage: python -m tools.run_tool_chain '<json_chain>'")
        return 2

    chain = _parse_chain(args[0])
    runner = Bootstrap.build_tool_chain_runner()
    outcome = runner.run(chain)
    _print_chain(outcome)

    while outcome.status == "confirmation_required" and outcome.confirmation_id:
        prompt = "Confirm? [s/N]: "
        if outcome.metadata is not None:
            prompt = str(outcome.metadata.get("prompt", prompt))

        response = input(prompt)
        outcome = runner.confirm(outcome.confirmation_id, response)
        _print_chain(outcome)

        if outcome.status == "invalid_confirmation":
            continue

        break

    return 0 if outcome.success else 1


def _parse_chain(raw: str) -> tuple[ToolChainStep, ...]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        print(f"Invalid JSON chain: {error}")
        raise SystemExit(2) from error

    raw_steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(raw_steps, list):
        print("Chain must be a JSON object with a steps list.")
        raise SystemExit(2)

    steps: list[ToolChainStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            print("Each chain step must be a JSON object.")
            raise SystemExit(2)

        arguments: Any = raw_step.get("arguments", {})
        if not isinstance(arguments, dict):
            print("Each chain step arguments value must be an object.")
            raise SystemExit(2)

        steps.append(
            ToolChainStep(
                step_id=str(raw_step.get("id", "")),
                tool_name=str(raw_step.get("tool_name", "")),
                arguments=arguments,
            )
        )

    return tuple(steps)


def _print_chain(outcome: ToolChainResult) -> None:
    print(f"Success: {str(outcome.success).lower()}")
    print(f"Status: {outcome.status}")
    print(f"Execution count: {outcome.execution_count}")

    if outcome.failed_step_id:
        print(f"Failed step: {outcome.failed_step_id}")

    if outcome.confirmation_id:
        print(f"Confirmation id: {outcome.confirmation_id}")

    for step in outcome.steps:
        print(f"Step: {step.step_id}")
        print(f"  Tool: {step.tool_name}")
        print(f"  Status: {step.result.status}")
        print(f"  Executed: {str(step.result.executed).lower()}")
        print(f"  Execution count: {step.result.execution_count}")

        if step.result.error_message:
            print(f"  Error: {step.result.error_message}")
        elif step.result.success:
            print(f"  Result: {_console_text(step.result.result)}")


def _console_text(value: Any) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


if __name__ == "__main__":
    raise SystemExit(main())
