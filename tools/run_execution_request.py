"""Manual CLI for conversational Atlas execution requests."""

from __future__ import annotations

import sys

from bootstrap.bootstrap import Bootstrap
from tools.execution_coordinator import ExecutionCoordinationResult
from use_cases.execution_result_presenter import ExecutionResultPresenter


def main(argv: list[str] | None = None) -> int:
    """Coordinate one natural-language execution request."""
    args = list(sys.argv[1:] if argv is None else argv)

    debug = False
    if "--debug" in args:
        debug = True
        args.remove("--debug")

    if len(args) != 1:
        print('Usage: python -m tools.run_execution_request [--debug] "<prompt>"')
        return 2

    coordinator = Bootstrap.build_execution_coordinator()
    presenter = ExecutionResultPresenter(debug=debug)
    result = coordinator.execute(args[0])
    _print_result(presenter, result, args[0])

    while result.confirmation_id is not None:
        response = input("Respuesta: ")
        result = coordinator.confirm(result.confirmation_id, response)
        _print_result(presenter, result, args[0])

        if result.status.value == "CONFIRMATION_REQUIRED":
            continue

        break

    return 0 if result.status.value in {"EXECUTED", "DIRECT_RESPONSE_REQUIRED"} else 1


def _print_result(
    presenter: ExecutionResultPresenter,
    result: ExecutionCoordinationResult,
    original_text: str,
) -> None:
    print(_console_text(presenter.present(result, original_text=original_text)))


def _console_text(value: object) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


if __name__ == "__main__":
    raise SystemExit(main())
