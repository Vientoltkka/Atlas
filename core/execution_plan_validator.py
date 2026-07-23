"""Validation for Atlas execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

from core.execution_arguments import (
    ExecutionArguments,
    InvalidExecutionArgumentError,
    contains_execution_variable_reference,
    contains_step_output_reference,
)
from core.execution_dependency_checker import (
    ImplicitStepDependencyError,
    TooManyStepDependenciesError,
)
from core.execution_plan_topology import (
    ExecutionPlanCycleError,
    ExecutionPlanTopologicalSorter,
    ExecutionPlanTopologyError,
)
from core.execution_condition import (
    AllOfCondition,
    AnyOfCondition,
    ExecutionCondition,
    InvalidConditionTreeError,
    NotCondition,
    is_execution_condition_node,
    iter_condition_operands,
    validate_condition_tree,
)
from core.execution_plan_output import (
    ExecutionPlanOutput,
    InvalidExecutionPlanOutputError,
)
from core.execution_plan_registry import (
    ExecutionPlanReference,
    ExecutionPlanRegistry,
    ExecutionPlanRegistryError,
)
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.parameter_resolver import (
    BLOCKED_REFERENCE_PARTS,
    MAX_TEMPLATE_LENGTH,
    MAX_TEMPLATE_REFERENCES,
    REFERENCE_PATTERN,
    TEMPLATE_REFERENCE_PATTERN,
)
from core.planner import ExecutionPlan
from core.step_output_reference import StepOutputReference
from tools.registry import ToolRegistry


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

    MAX_STEP_DEPENDENCIES = 64
    MAX_SUBPLAN_DEPTH = 8
    _VALID_PLAN_STATUSES = {"planned"}
    _VALID_STEP_STATUSES = {"pending"}
    _DANGEROUS_TOOLS = {
        "write_file",
        "desktop.type_text",
        "desktop.press_hotkey",
    }
    _TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        topological_sorter: ExecutionPlanTopologicalSorter | None = None,
        plan_registry: ExecutionPlanRegistry | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._topological_sorter = topological_sorter or ExecutionPlanTopologicalSorter()
        self._plan_registry = plan_registry

    def validate(
        self,
        plan: ExecutionPlan,
        *,
        depth: int = 0,
        plan_stack: tuple[int, ...] = (),
        reference_stack: tuple[ExecutionPlanReference, ...] = (),
    ) -> PlanValidationResult:
        """Return a structured validation result for an execution plan."""
        errors: list[str] = []
        warnings: list[str] = []
        self._validate_plan(
            plan,
            errors,
            warnings,
            depth=depth,
            plan_stack=plan_stack,
            reference_stack=reference_stack,
        )

        is_valid = not errors

        return PlanValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            requires_confirmation=plan.requires_confirmation,
            status="valid" if is_valid else "invalid",
            plan_signature=plan_signature(plan) if is_valid else None,
        )

    def _validate_plan(
        self,
        plan: ExecutionPlan,
        errors: list[str],
        warnings: list[str],
        *,
        depth: int,
        plan_stack: tuple[int, ...],
        reference_stack: tuple[ExecutionPlanReference, ...],
    ) -> None:
        if not isinstance(plan, ExecutionPlan):
            errors.append("Subplan must be an ExecutionPlan.")
            return
        if depth > self.MAX_SUBPLAN_DEPTH:
            errors.append(
                f"SubplanDepthExceededError: subplan depth {depth} exceeds "
                f"maximum {self.MAX_SUBPLAN_DEPTH}."
            )
            return
        plan_identity = id(plan)
        if plan_identity in plan_stack:
            errors.append("RecursiveSubplanError: recursive subplan reference detected.")
            return
        active_stack = plan_stack + (plan_identity,)
        self._validate_goal(plan, errors)
        self._validate_steps_presence(plan, errors)
        self._validate_statuses(plan, errors)
        self._validate_estimated_steps(plan, errors)
        self._validate_step_ids(plan, errors)
        self._validate_step_actions(plan, errors)
        self._validate_tools(plan, errors)
        self._validate_arguments(plan, errors)
        self._validate_tool_argument_schemas(plan, errors)
        self._validate_dependencies(plan, errors)
        self._validate_topology(plan, errors)
        self._validate_plan_output(plan, errors)
        self._validate_confirmation(plan, errors, warnings)
        self._validate_warnings(plan, warnings)
        self._validate_subplans(
            plan,
            errors,
            warnings,
            depth=depth,
            plan_stack=active_stack,
            reference_stack=reference_stack,
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
            if step.subplan is not None or getattr(step, "subplan_ref", None) is not None:
                continue
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

            if (
                step.tool is None
                and step.subplan is None
                and getattr(step, "subplan_ref", None) is None
                and step.arguments
            ):
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
                validate_source = (
                    step.arguments.as_dict()
                    if isinstance(step.arguments, ExecutionArguments)
                    else step.arguments
                )
                ExecutionArguments(validate_source)
                _signature_safe_value(validate_source)
            except (InvalidExecutionArgumentError, TypeError) as error:
                errors.append(f"Step '{step.id}' arguments are invalid: {error}.")

            self._validate_output_binding(step, errors)
            self._validate_condition(step, errors)

        self._validate_static_references(plan, errors)
        self._validate_structured_references(plan, errors)

    def _validate_condition(
        self,
        step: Any,
        errors: list[str],
    ) -> None:
        condition = getattr(step, "condition", None)
        if condition is None:
            return
        if not is_execution_condition_node(condition):
            errors.append(f"Step '{step.id}' condition must be a valid condition node.")
            return
        try:
            validate_condition_tree(condition, step_id=step.id)
            _signature_safe_value(condition)
        except (InvalidConditionTreeError, TypeError) as error:
            errors.append(f"Step '{step.id}' condition is invalid: {error}.")

    def _validate_output_binding(
        self,
        step: Any,
        errors: list[str],
    ) -> None:
        binding = getattr(step, "output_binding", None)
        if binding is None:
            return
        if not isinstance(binding, ExecutionVariableBinding):
            errors.append(f"Step '{step.id}' output_binding must be ExecutionVariableBinding.")
            return
        if type(binding.overwrite) is not bool:
            errors.append(f"Step '{step.id}' output_binding overwrite must be boolean.")

    def _validate_tool_argument_schemas(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        if self._tool_registry is None:
            return

        for step in plan.ordered_steps:
            if step.subplan is not None or getattr(step, "subplan_ref", None) is not None:
                continue
            if step.tool in {None, "direct_response"}:
                continue

            assert step.tool is not None
            if not self._tool_registry.exists(step.tool):
                continue

            schema = self._tool_registry.arguments_schema(step.tool)
            if schema is None:
                continue

            arguments = (
                step.arguments.as_dict()
                if isinstance(step.arguments, ExecutionArguments)
                else dict(step.arguments)
            )
            validation = schema.validate(step.tool, arguments)
            if validation.is_valid:
                continue

            for error in validation.errors:
                if (
                    error.parameter_name is not None
                    and error.parameter_name in arguments
                    and self._is_deferred_argument(arguments[error.parameter_name])
                ):
                    continue
                errors.append(
                    f"Step '{step.id}' schema validation failed: {error.message}."
                )

    def _is_deferred_argument(
        self,
        value: Any,
    ) -> bool:
        return (
            contains_step_output_reference(value)
            or contains_execution_variable_reference(value)
            or any(
            self._iter_special_objects(value, key)
            for key in ("$ref", "$template")
        )
        )

    def _validate_structured_references(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        ordered_ids = [step.id for step in plan.ordered_steps]
        first_index_by_id: dict[str, int] = {}
        for index, step_id in enumerate(ordered_ids):
            first_index_by_id.setdefault(step_id, index)

        for index, step in enumerate(plan.ordered_steps):
            reference_sources: list[Any] = [step.arguments]
            if getattr(step, "condition", None) is not None:
                reference_sources.extend(iter_condition_operands(step.condition))
            for reference_source in reference_sources:
                for reference in self._iter_structured_references(reference_source):
                    if isinstance(reference, StepOutputReference):
                        referenced = reference.step_id
                        if referenced not in first_index_by_id:
                            errors.append(
                                f"Step '{step.id}' references unknown step '{referenced}'."
                            )
                            continue
                        if referenced == step.id:
                            errors.append(f"Step '{step.id}' cannot reference itself.")
                            continue
                        if referenced not in step.depends_on:
                            errors.append(
                                f"{ImplicitStepDependencyError.__name__}: "
                                f"Step '{step.id}' references step '{referenced}' "
                                "without declaring it in depends_on."
                            )
                        path = reference.path
                    else:
                        path = reference.path

                    for segment in path:
                        if isinstance(segment, str) and segment in BLOCKED_REFERENCE_PARTS:
                            errors.append(
                                f"Step '{step.id}' has unsafe reference path segment: {segment}."
                            )
    def _iter_structured_references(
        self,
        value: Any,
    ) -> tuple[StepOutputReference | ExecutionVariableReference, ...]:
        references: list[StepOutputReference | ExecutionVariableReference] = []

        def visit(item: Any) -> None:
            if isinstance(item, (StepOutputReference, ExecutionVariableReference)):
                references.append(item)
                return

            if isinstance(item, Mapping):
                for nested in item.values():
                    visit(nested)
                return

            if isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)
        return tuple(references)

    def _validate_static_references(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        step_ids = {step.id for step in plan.ordered_steps}
        dependencies_by_step = {
            step.id: tuple(step.dependencies)
            for step in plan.ordered_steps
        }

        for step in plan.ordered_steps:
            allowed_references = self._transitive_dependencies(
                step.id,
                dependencies_by_step,
            )
            reference_sources: list[Any] = [step.arguments]
            if getattr(step, "condition", None) is not None:
                reference_sources.extend(iter_condition_operands(step.condition))

            for reference_source in reference_sources:
                for reference in self._iter_special_objects(reference_source, "$ref"):
                    ref_keys = tuple(reference.keys())
                    if ref_keys != ("$ref",):
                        errors.append(
                            f"Step '{step.id}' reference objects must contain only '$ref'."
                        )
                        continue

                    raw_reference = reference["$ref"]
                    if not isinstance(raw_reference, str) or not raw_reference.strip():
                        errors.append(
                            f"Step '{step.id}' reference value must be a non-empty string."
                        )
                        continue

                    ref_value = raw_reference.strip()
                    match = REFERENCE_PATTERN.fullmatch(ref_value)
                    if match is None:
                        errors.append(
                            f"Step '{step.id}' has invalid reference syntax: {ref_value}."
                        )
                        continue

                    referenced_step_id = match.group(1)
                    ref_path = match.group(2)

                    self._validate_reference_parts(step.id, ref_path, errors)
                    self._validate_reference_dependency(
                        step.id,
                        referenced_step_id,
                        step_ids,
                        allowed_references,
                        errors,
                    )

                for template in self._iter_special_objects(reference_source, "$template"):
                    template_keys = tuple(template.keys())
                    if template_keys != ("$template",):
                        errors.append(
                            f"Step '{step.id}' template objects must contain only '$template'."
                        )
                        continue

                    raw_template = template["$template"]
                    if not isinstance(raw_template, str):
                        errors.append(
                            f"Step '{step.id}' template value must be a string."
                        )
                        continue

                    if len(raw_template) > MAX_TEMPLATE_LENGTH:
                        errors.append(
                            f"Step '{step.id}' template exceeds the maximum supported length."
                        )
                        continue

                    escaped_template = (
                        raw_template
                        .replace("{{{{", "\u0000ATLAS_OPEN_BRACE\u0000")
                        .replace("}}}}", "\u0000ATLAS_CLOSE_BRACE\u0000")
                    )
                    template_references = list(TEMPLATE_REFERENCE_PATTERN.finditer(escaped_template))

                    if len(template_references) > MAX_TEMPLATE_REFERENCES:
                        errors.append(
                            f"Step '{step.id}' template exceeds the maximum number of references."
                        )

                    remainder = TEMPLATE_REFERENCE_PATTERN.sub("", escaped_template)
                    if "{{" in remainder or "}}" in remainder:
                        errors.append(
                            f"Step '{step.id}' has invalid template brace syntax."
                        )
                        continue

                    for template_reference in template_references:
                        expression = template_reference.group(1).strip()
                        match = REFERENCE_PATTERN.fullmatch(expression)
                        if match is None:
                            if any(
                                token in expression
                                for token in ("(", ")", "+", "-", "*", "/", "[", "]", "|", "=", "<", ">")
                            ):
                                errors.append(
                                    f"Step '{step.id}' has unsupported template expression: {expression}."
                                )
                            else:
                                errors.append(
                                    f"Step '{step.id}' has invalid template reference syntax: {expression}."
                                )
                            continue

                        referenced_step_id = match.group(1)
                        ref_path = match.group(2)
                        self._validate_reference_parts(step.id, ref_path, errors)
                        self._validate_reference_dependency(
                            step.id,
                            referenced_step_id,
                            step_ids,
                            allowed_references,
                            errors,
                        )

    def _iter_special_objects(
        self,
        value: Any,
        key: str,
    ) -> tuple[Mapping[str, Any], ...]:
        references: list[Mapping[str, Any]] = []

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                if key in item:
                    references.append(item)
                    return

                for nested in item.values():
                    visit(nested)
                return

            if isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)
        return tuple(references)

    def _validate_reference_parts(
        self,
        step_id: str,
        ref_path: str | None,
        errors: list[str],
    ) -> None:
        if ref_path is None:
            return

        for part in ref_path.split("."):
            if (
                not part
                or part.startswith("_")
                or part in BLOCKED_REFERENCE_PARTS
            ):
                errors.append(
                    f"Step '{step_id}' has unsafe reference path segment: {part}."
                )

    def _validate_reference_dependency(
        self,
        step_id: str,
        referenced_step_id: str,
        step_ids: set[str],
        allowed_references: set[str],
        errors: list[str],
    ) -> None:
        if referenced_step_id not in step_ids:
            errors.append(
                f"Step '{step_id}' references unknown step '{referenced_step_id}'."
            )
            return

        if referenced_step_id == step_id:
            errors.append(f"Step '{step_id}' cannot reference itself.")
            return

        if referenced_step_id not in allowed_references:
            errors.append(
                f"Step '{step_id}' references non-dependent step '{referenced_step_id}'."
            )

    def _transitive_dependencies(
        self,
        step_id: str,
        dependencies_by_step: dict[str, tuple[str, ...]],
    ) -> set[str]:
        dependencies: set[str] = set()
        pending = list(dependencies_by_step.get(step_id, ()))

        while pending:
            dependency = pending.pop()
            if dependency in dependencies:
                continue

            dependencies.add(dependency)
            pending.extend(dependencies_by_step.get(dependency, ()))

        return dependencies

    def _validate_dependencies(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        step_ids = [step.id for step in plan.ordered_steps]
        unique_ids = set(step_ids)
        dependency_graph: dict[str, tuple[str, ...]] = {}

        for step in plan.ordered_steps:
            dependencies = tuple(step.depends_on)
            dependency_graph[step.id] = tuple(
                dependency
                for dependency in dependencies
                if isinstance(dependency, str)
            )

            if len(dependencies) > self.MAX_STEP_DEPENDENCIES:
                errors.append(
                    f"{TooManyStepDependenciesError.__name__}: "
                    f"Step '{step.id}' declares {len(dependencies)} dependencies; "
                    f"maximum is {self.MAX_STEP_DEPENDENCIES}."
                )

            seen_dependencies: set[str] = set()
            for position, dependency in enumerate(dependencies):
                if not isinstance(dependency, str):
                    errors.append(
                        f"Step '{step.id}' dependency at position {position} "
                        "must be a string."
                    )
                    continue

                if not dependency.strip():
                    errors.append(
                        f"Step '{step.id}' dependency at position {position} "
                        "cannot be empty."
                    )
                    continue

                if dependency in seen_dependencies:
                    errors.append(
                        f"Step '{step.id}' declares duplicate dependency "
                        f"'{dependency}' at position {position}."
                    )
                seen_dependencies.add(dependency)

                if dependency not in unique_ids:
                    errors.append(
                        f"Step '{step.id}' depends on unknown step '{dependency}'."
                    )

                if dependency == step.id:
                    errors.append(f"Step '{step.id}' cannot depend on itself.")

        for cycle in self._find_cycles(dependency_graph):
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}.")

    def _validate_topology(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        try:
            self._topological_sorter.sort(plan)
        except ExecutionPlanCycleError as error:
            errors.append(f"ExecutionPlanCycleError: {error}.")
        except ExecutionPlanTopologyError as error:
            errors.append(f"ExecutionPlanTopologyError: {error}.")
        except ValueError as error:
            errors.append(f"ExecutionPlanTopologyError: {error}.")

    def _validate_confirmation(
        self,
        plan: ExecutionPlan,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        dangerous_tools = tuple(
            step.tool
            for step in plan.ordered_steps
            if step.subplan is None
            and getattr(step, "subplan_ref", None) is None
            and step.tool is not None
            and step.tool in self._DANGEROUS_TOOLS
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
            if step.subplan is None
            and getattr(step, "subplan_ref", None) is None
            and step.tool is not None
            and step.tool != "direct_response"
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

    def _validate_step_actions(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        for step in plan.ordered_steps:
            has_tool = step.tool is not None
            has_subplan = step.subplan is not None
            has_subplan_ref = getattr(step, "subplan_ref", None) is not None
            if sum((has_tool, has_subplan, has_subplan_ref)) != 1:
                errors.append(
                    f"InvalidSubplanStepError: Step '{step.id}' must define "
                    "exactly one of tool, subplan, or subplan_ref."
                )
            if step.subplan is not None and not isinstance(step.subplan, ExecutionPlan):
                errors.append(
                    f"InvalidSubplanStepError: Step '{step.id}' subplan must be "
                    "an ExecutionPlan."
                )
            if (
                getattr(step, "subplan_ref", None) is not None
                and not isinstance(step.subplan_ref, ExecutionPlanReference)
            ):
                errors.append(
                    f"InvalidExecutionPlanReferenceError: Step '{step.id}' "
                    "subplan_ref must be ExecutionPlanReference."
                )

    def _validate_subplans(
        self,
        plan: ExecutionPlan,
        errors: list[str],
        warnings: list[str],
        *,
        depth: int,
        plan_stack: tuple[int, ...],
        reference_stack: tuple[ExecutionPlanReference, ...],
    ) -> None:
        for step in plan.ordered_steps:
            if step.subplan is not None:
                if not isinstance(step.subplan, ExecutionPlan):
                    continue
                self._validate_plan(
                    step.subplan,
                    errors,
                    warnings,
                    depth=depth + 1,
                    plan_stack=plan_stack,
                    reference_stack=reference_stack,
                )
                continue

            reference = getattr(step, "subplan_ref", None)
            if reference is None:
                continue
            if not isinstance(reference, ExecutionPlanReference):
                continue
            if reference in reference_stack:
                errors.append(
                    "RecursiveRegisteredExecutionPlanError: recursive registered "
                    f"plan reference detected for '{reference.plan_id}'."
                )
                continue
            if self._plan_registry is None:
                errors.append(
                    f"ExecutionPlanRegistryUnavailableError: Step '{step.id}' "
                    "uses subplan_ref but no ExecutionPlanRegistry was injected."
                )
                continue
            try:
                resolved_plan = self._plan_registry.resolve(reference)
            except ExecutionPlanRegistryError as error:
                errors.append(
                    f"{type(error).__name__}: Step '{step.id}' cannot resolve "
                    f"registered plan '{reference.plan_id}'."
                )
                continue
            if not isinstance(resolved_plan, ExecutionPlan):
                errors.append(
                    f"RegisteredExecutionPlanValidationError: Step '{step.id}' "
                    "resolved plan is not an ExecutionPlan."
                )
                continue
            self._validate_plan(
                resolved_plan,
                errors,
                warnings,
                depth=depth + 1,
                plan_stack=plan_stack,
                reference_stack=reference_stack + (reference,),
            )

    def _validate_plan_output(
        self,
        plan: ExecutionPlan,
        errors: list[str],
    ) -> None:
        if plan.output is None:
            return
        if not isinstance(plan.output, ExecutionPlanOutput):
            errors.append("ExecutionPlan output must be ExecutionPlanOutput.")
            return
        try:
            definition = plan.output.as_definition()
            _signature_safe_value(definition)
        except (InvalidExecutionPlanOutputError, TypeError) as error:
            errors.append(f"ExecutionPlan output is invalid: {error}.")
            return

        step_ids = {step.id for step in plan.ordered_steps}
        for reference in self._iter_structured_references(definition):
            if isinstance(reference, StepOutputReference):
                if reference.step_id not in step_ids:
                    errors.append(
                        f"ExecutionPlan output references unknown step '{reference.step_id}'."
                    )
                path = reference.path
            else:
                path = reference.path
            for segment in path:
                if isinstance(segment, str) and segment in BLOCKED_REFERENCE_PARTS:
                    errors.append(
                        "ExecutionPlan output has unsafe reference path segment: "
                        f"{segment}."
                    )

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
                "subplan": _signature_safe_value(step.subplan),
                "subplan_ref": _signature_safe_value(getattr(step, "subplan_ref", None)),
                "depends_on": list(step.depends_on),
                "status": step.status,
                "arguments": _signature_safe_value(
                    step.arguments.as_dict()
                    if isinstance(step.arguments, ExecutionArguments)
                    else dict(step.arguments)
                ),
                "output_binding": _signature_safe_value(step.output_binding),
                "condition": _signature_safe_value(step.condition),
            }
            for step in plan.ordered_steps
        ],
        "estimated_steps": plan.estimated_steps,
        "required_tools": list(plan.required_tools),
        "detected_risks": list(plan.detected_risks),
        "requires_confirmation": plan.requires_confirmation,
        "status": plan.status,
        "output": _signature_safe_value(
            plan.output.as_definition()
            if isinstance(plan.output, ExecutionPlanOutput)
            else None
        ),
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
    if isinstance(value, ExecutionPlan):
        return {
            "$type": "execution_plan",
            "goal": value.goal,
            "ordered_steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "tool": step.tool,
                    "subplan": _signature_safe_value(step.subplan),
                    "subplan_ref": _signature_safe_value(getattr(step, "subplan_ref", None)),
                    "depends_on": list(step.depends_on),
                    "status": step.status,
                    "arguments": _signature_safe_value(
                        step.arguments.as_dict()
                        if isinstance(step.arguments, ExecutionArguments)
                        else dict(step.arguments)
                    ),
                    "output_binding": _signature_safe_value(step.output_binding),
                    "condition": _signature_safe_value(step.condition),
                }
                for step in value.ordered_steps
            ],
            "estimated_steps": value.estimated_steps,
            "required_tools": list(value.required_tools),
            "detected_risks": list(value.detected_risks),
            "requires_confirmation": value.requires_confirmation,
            "status": value.status,
            "output": _signature_safe_value(
                value.output.as_definition()
                if isinstance(value.output, ExecutionPlanOutput)
                else None
            ),
        }

    if isinstance(value, ExecutionPlanOutput):
        return {
            "$type": "execution_plan_output",
            "value": _signature_safe_value(value.as_definition()),
        }

    if isinstance(value, ExecutionPlanReference):
        return {
            "$type": "execution_plan_reference",
            "plan_id": value.plan_id,
            "version": value.version,
        }

    if isinstance(value, StepOutputReference):
        return {
            "$type": "step_output_reference",
            "step_id": value.step_id,
            "path": list(value.path),
        }

    if isinstance(value, ExecutionVariableReference):
        return {
            "$type": "execution_variable_reference",
            "name": value.name,
            "path": list(value.path),
        }

    if isinstance(value, ExecutionVariableBinding):
        return {
            "$type": "execution_variable_binding",
            "variable_name": value.variable_name,
            "path": list(value.path),
            "overwrite": value.overwrite,
        }

    if isinstance(value, ExecutionCondition):
        return {
            "$type": "execution_condition",
            "operator": value.operator.value,
            "left": _signature_safe_value(value.left),
            "right": _signature_safe_value(value.right),
        }

    if isinstance(value, AllOfCondition):
        return {
            "$type": "all_of_condition",
            "conditions": [_signature_safe_value(item) for item in value.conditions],
        }

    if isinstance(value, AnyOfCondition):
        return {
            "$type": "any_of_condition",
            "conditions": [_signature_safe_value(item) for item in value.conditions],
        }

    if isinstance(value, NotCondition):
        return {
            "$type": "not_condition",
            "condition": _signature_safe_value(value.condition),
        }

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite float values are not supported")
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
