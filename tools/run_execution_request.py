"""Manual CLI for coordinated Atlas execution requests."""

from __future__ import annotations

import json
import sys
from typing import Any

from bootstrap.bootstrap import Bootstrap
from tools.execution_coordinator import ExecutionCoordinationResult


def main(argv: list[str] | None = None) -> int:
    """Coordinate one natural-language execution request."""
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) != 1:
        print('Usage: python -m tools.run_execution_request "<prompt>"')
        return 2

    coordinator = Bootstrap.build_execution_coordinator()
    result = coordinator.execute(args[0])
    _print_result(result)

    while result.confirmation_id is not None:
        response = input("Confirm? [s/N]: ")
        result = coordinator.confirm(result.confirmation_id, response)
        _print_result(result)

        if result.status.value == "CONFIRMATION_REQUIRED":
            continue

        break

    return 0 if result.status.value in {"EXECUTED", "DIRECT_RESPONSE_REQUIRED"} else 1


def _print_result(result: ExecutionCoordinationResult) -> None:
    print(f"Status: {result.status.value}")
    print(f"Mode: {result.mode.value}")
    print(f"Executed: {str(result.executed).lower()}")
    print(f"Message: {result.message}")

    if result.confirmation_id:
        print(f"Confirmation id: {result.confirmation_id}")

    if result.missing_information:
        print(f"Missing: {', '.join(result.missing_information)}")

    if result.ambiguous_information:
        print(f"Ambiguous: {', '.join(result.ambiguous_information)}")

    if result.validation_errors:
        print(f"Validation errors: {', '.join(result.validation_errors)}")

    if result.execution_result is not None:
        print(
            "Execution result: "
            + json.dumps(
                _execution_summary(result.execution_result),
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def _execution_summary(execution_result: Any) -> dict[str, Any]:
    summary = {
        "status": getattr(execution_result, "status", None),
        "success": getattr(execution_result, "success", None),
        "execution_count": getattr(execution_result, "execution_count", None),
    }

    tool_name = getattr(execution_result, "tool_name", None)
    if tool_name is not None:
        summary["tool_name"] = tool_name

    failed_step_id = getattr(execution_result, "failed_step_id", None)
    if failed_step_id is not None:
        summary["failed_step_id"] = failed_step_id

    return summary


if __name__ == "__main__":
    raise SystemExit(main())
