"""Safe manual probe for structured hybrid planning.

This script plans only. It does not execute tools or start the Atlas orchestrator.
"""

from __future__ import annotations

import argparse
import json

from bootstrap.bootstrap import Bootstrap
from core.deterministic_multi_tool_planner import DeterministicMultiToolPlanner
from core.hybrid_execution_planner import HybridExecutionPlanner


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a structured execution plan without executing tools.")
    parser.add_argument("objective", help="Objective to plan.")
    args = parser.parse_args()

    registry = Bootstrap.build_tool_registry()
    selector = Bootstrap.build_tool_selector(registry)
    schemas = Bootstrap.build_argument_schema_registry()
    catalog = Bootstrap.build_semantic_tool_catalog(registry, selector, schemas)
    hybrid_planner: HybridExecutionPlanner = Bootstrap.build_hybrid_execution_planner(
        registry,
        schemas,
    )
    provider = Bootstrap.build_structured_plan_provider()

    result = hybrid_planner.plan(
        args.objective,
        deterministic_planner=DeterministicMultiToolPlanner(),
        catalog=catalog,
        selector=selector,
        plan_provider=provider,
    )

    payload = {
        "success": result.success,
        "handled": result.handled,
        "source": result.source,
        "error_code": result.error_code,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "requires_clarification": result.requires_clarification,
        "missing_information": list(result.missing_information),
        "validation": (
            {
                "is_valid": result.validation_result.is_valid,
                "status": result.validation_result.status,
                "errors": result.validation_result.errors,
                "warnings": result.validation_result.warnings,
                "requires_confirmation": result.validation_result.requires_confirmation,
            }
            if result.validation_result is not None
            else None
        ),
        "plan": _plan_to_dict(result.plan),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _plan_to_dict(plan) -> dict | None:
    if plan is None:
        return None
    return {
        "goal": plan.goal,
        "estimated_steps": plan.estimated_steps,
        "required_tools": list(plan.required_tools),
        "detected_risks": list(plan.detected_risks),
        "requires_confirmation": plan.requires_confirmation,
        "status": plan.status,
        "steps": [
            {
                "id": step.id,
                "description": step.description,
                "tool": step.tool,
                "dependencies": list(step.dependencies),
                "status": step.status,
                "arguments": dict(step.arguments),
            }
            for step in plan.ordered_steps
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
