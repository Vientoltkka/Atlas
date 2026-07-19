"""Validation for Atlas execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

from core.planner import ExecutionPlan


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    """Structured result for execution plan validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    status: str = "invalid"
    plan_signature: str | None = None


class ExecutionPlanValidator:
    """Validate execution plans without executing or modifying them."""

    _VALID_PLAN_STATUSES = {"planned"}
    _VALID_STEP_STATUSES = {"pending"}
    _DANGEROUS_TOOLS = {
        "write_file",
        "desktop.type_text",
        "desktop.press_hotkey",
    }
    _TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

    def validate(
        self,
        plan: ExecutionPlan,
    ) -> PlanValidationResult:
        """Return a structured validation result for an execution plan."""
        errors: list[str] = []
        warnings: list[str] = []

        self._validate_goal(plan, errors)
        self._validate_steps_presence(plan, errors)
        self._validate_statuses(plan, errors)
        self._validate_estimated_steps(plan, errors)
        self._validate_step_ids(plan, errors)
        self._validate_tools(plan, errors)
        self._validate_arguments(plan, errors)
        self._validate_dependencies(plan, errors)
        self._validate_confirmation(plan, errors, warnings)
        self._validate_warnings(plan, warnings)

        is_valid = not errors

        return PlanValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            requires_confirmation=plan.requires_confirmation,
            status="valid" if is_valid else "invalid",
            plan_signature=plan_signature(plan) if is_valid else None,
        )

    def _validate_goal(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        if not plan.goal.strip():
            errors.append("Plan goal cannot be empty.")

    def _validate_steps_presence(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        if not plan.ordered_steps:
            errors.append("Plan must contain at least one step.")

    def _validate_statuses(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        if plan.status not in self._VALID_PLAN_STATUSES:
            errors.append(f"Invalid initial plan status: {plan.status}.")

        for step in plan.ordered_steps:
            if step.status not in self._VALID_STEP_STATUSES:
                errors.append(
                    f"Invalid initial status for step '{step.id}': {step.status}."
                )

    def _validate_estimated_steps(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        if plan.estimated_steps != len(plan.ordered_steps):
            errors.append(
                "Plan estimated_steps must match the real number of ordered steps."
            )

    def _validate_step_ids(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        ids = [step.id for step in plan.ordered_steps]

        if any(not step_id.strip() for step_id in ids):
            errors.append("Step ids cannot be empty.")

        duplicates = sorted(
            {
                step_id
                for step_id in ids
                if ids.count(step_id) > 1
            }
        )

        for step_id in duplicates:
            errors.append(f"Duplicate step id: {step_id}.")

    def _validate_tools(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        declared_tools = set(plan.required_tools)

        for required_tool in plan.required_tools:
            if not self._is_well_formed_tool(required_tool):
                errors.append(f"Malformed required tool: {required_tool}.")

        for step in plan.ordered_steps:
            if step.tool is None:
                continue

            if not self._is_well_formed_tool(step.tool):
                errors.append(f"Malformed tool for step '{step.id}': {step.tool}.")
                continue

            if step.tool != "direct_response" and step.tool not in declared_tools:
                errors.append(
                    f"Step '{step.id}' uses undeclared tool '{step.tool}'."
                )

    def _validate_arguments(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        for step in plan.ordered_steps:
            if not isinstance(step.arguments, Mapping):
                errors.append(f"Step '{step.id}' arguments must be a mapping.")
                continue

            if step.tool is None and step.arguments:
                errors.append(
                    f"Logical step '{step.id}' cannot declare arguments."
                )

            for key in step.arguments:
                if not isinstance(key, str):
                    errors.append(
                        f"Step '{step.id}' argument keys must be strings."
                    )
                    continue

                if not key.strip():
                    errors.append(
                        f"Step '{step.id}' argument keys cannot be empty."
                    )

            try:
                _signature_safe_value(dict(step.arguments))
            except TypeError as error:
                errors.append(
                    f"Step '{step.id}' arguments are not deterministically serializable: {error}."
                )

    def _validate_dependencies(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        step_ids = [step.id for step in plan.ordered_steps]
        unique_ids = set(step_ids)
        seen_ids: set[str] = set()
        dependency_graph: dict[str, tuple[str, ...]] = {}

        for step in plan.ordered_steps:
            dependency_graph[step.id] = tuple(step.dependencies)

            for dependency in step.dependencies:
                if dependency not in unique_ids:
                    errors.append(
                        f"Step '{step.id}' depends on unknown step '{dependency}'."
                    )

                if dependency == step.id:
                    errors.append(f"Step '{step.id}' cannot depend on itself.")

                if dependency in unique_ids and dependency not in seen_ids:
                    errors.append(
                        f"Step '{step.id}' depends on '{dependency}' before it is executable."
                    )

            seen_ids.add(step.id)

        for cycle in self._find_cycles(dependency_graph):
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}.")

    def _validate_confirmation(
        self,
        plan: ExecutionPlan,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        dangerous_tools = tuple(
            step.tool
            for step in plan.ordered_steps
            if step.tool is not None and step.tool in self._DANGEROUS_TOOLS
        )

        if dangerous_tools and not plan.requires_confirmation:
            errors.append("Dangerous plan cannot be marked as not requiring confirmation.")

        if plan.requires_confirmation and not dangerous_tools:
            warnings.append(
                "Plan requires confirmation but no confirmation-gated tool was detected."
            )

    def _validate_warnings(
        self,
        plan: ExecutionPlan,
        warnings: list[str],
    ) -> None:
        if plan.detected_risks and not plan.requires_confirmation:
            warnings.append(
                "Plan declares risks but does not require confirmation."
            )

        used_tools = {
            step.tool
            for step in plan.ordered_steps
            if step.tool is not None and step.tool != "direct_response"
        }

        for required_tool in plan.required_tools:
            if required_tool not in used_tools:
                warnings.append(
                    f"Required tool '{required_tool}' is declared but not used by any step."
                )

    def _is_well_formed_tool(
        self,
        tool: str,
    ) -> bool:
        return bool(tool.strip()) and self._TOOL_NAME_PATTERN.fullmatch(tool) is not None

    def _find_cycles(
        self,
        dependency_graph: dict[str, tuple[str, ...]],
    ) -> tuple[tuple[str, ...], ...]:
        cycles: list[tuple[str, ...]] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(
            step_id: str,
            path: tuple[str, ...],
        ) -> None:
            if step_id in visiting:
                cycle_start = path.index(step_id)
                cycles.append(path[cycle_start:] + (step_id,))
                return

            if step_id in visited:
                return

            visiting.add(step_id)

            for dependency in dependency_graph.get(step_id, ()):
                if dependency in dependency_graph:
                    visit(dependency, path + (dependency,))

            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in dependency_graph:
            visit(step_id, (step_id,))

        return tuple(cycles)


def plan_signature(
    plan: ExecutionPlan,
) -> str:
    """Return a deterministic signature for a plan's executable structure."""
    payload = {
        "goal": plan.goal,
        "ordered_steps": [
            {
                "id": step.id,
                "description": step.description,
                "tool": step.tool,
                "dependencies": list(step.dependencies),
                "status": step.status,
                "arguments": _signature_safe_value(dict(step.arguments)),
            }
            for step in plan.ordered_steps
        ],
        "estimated_steps": plan.estimated_steps,
        "required_tools": list(plan.required_tools),
        "detected_risks": list(plan.detected_risks),
        "requires_confirmation": plan.requires_confirmation,
        "status": plan.status,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _signature_safe_value(
    value: Any,
) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_signature_safe_value(item) for item in value]

    if isinstance(value, tuple):
        return [_signature_safe_value(item) for item in value]

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("mapping keys must be strings")

            if not key.strip():
                raise TypeError("mapping keys cannot be empty")

            normalized[key] = _signature_safe_value(item)

        return normalized

    raise TypeError(f"unsupported value type {type(value).__name__}")
