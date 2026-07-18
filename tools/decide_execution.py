"""Manual inspection for Atlas execution-mode decisions."""

from __future__ import annotations

import sys

from bootstrap.bootstrap import Bootstrap
from tools.execution_decision import ExecutionDecision


def main(argv: list[str] | None = None) -> int:
    """Print a structured execution decision for one prompt."""
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) != 1:
        print('Usage: python -m tools.decide_execution "<prompt>"')
        return 2

    engine = Bootstrap.build_execution_decision_engine()
    decision = engine.decide(args[0])
    _print_decision(decision)

    return 0


def _print_decision(decision: ExecutionDecision) -> None:
    print(f"Mode: {decision.mode.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reason: {decision.reason}")
    print(f"Candidate tools: {', '.join(decision.candidate_tools)}")
    print(f"Required capabilities: {', '.join(decision.required_capabilities)}")


if __name__ == "__main__":
    raise SystemExit(main())
