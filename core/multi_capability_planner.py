"""Deterministic planning across multiple workflow capabilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from core.capability_resolver import (
    CapabilityDefinition,
    CapabilityType,
    WorkflowCapabilityProvider,
    WorkflowCapabilitySource,
)
from core.execution_plan_library import ExecutionPlanLibrary, WorkflowDefinition
from core.execution_plan_output import ExecutionPlanOutput
from core.execution_variable_binding import ExecutionVariableBinding
from core.execution_variable_reference import ExecutionVariableReference
from core.planner import ExecutionPlan, ExecutionStep


MAX_MULTI_CAPABILITY_STEPS = 16


class MultiCapabilityPlanningError(RuntimeError):
    """Base error for deterministic multi-capability planning."""


class InvalidMultiCapabilityPlanningRequestError(MultiCapabilityPlanningError):
    """Raised when a multi-capability planning request is malformed."""


class MultiCapabilityPlanningStatus(str, Enum):
    """Stable outcomes for deterministic multi-capability planning."""

    PLANNED = "planned"
    INVALID_REQUEST = "invalid_request"
    IMPOSSIBLE_DEPENDENCY = "impossible_dependency"
    CYCLE_DETECTED = "cycle_detected"
    AMBIGUOUS_GRAPH = "ambiguous_graph"


@dataclass(frozen=True, slots=True)
class MultiCapabilityPlanningRequest:
    """Requested inputs and outputs for a composed workflow capability plan."""

    initial_inputs: tuple[str, ...]
    required_outputs: tuple[str, ...]
    required_capability_ids: tuple[str, ...] = ()
    maximum_steps: int = MAX_MULTI_CAPABILITY_STEPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_inputs", _validate_names(self.initial_inputs, "initial_inputs"))
        object.__setattr__(self, "required_outputs", _validate_names(self.required_outputs, "required_outputs"))
        object.__setattr__(
            self,
            "required_capability_ids",
            _validate_names(self.required_capability_ids, "required_capability_ids", dotted=True),
        )
        if isinstance(self.maximum_steps, bool) or not isinstance(self.maximum_steps, int):
            raise InvalidMultiCapabilityPlanningRequestError("maximum_steps must be an int.")
        if self.maximum_steps <= 0 or self.maximum_steps > MAX_MULTI_CAPABILITY_STEPS:
            raise InvalidMultiCapabilityPlanningRequestError("maximum_steps is outside the allowed range.")


@dataclass(frozen=True, slots=True)
class MultiCapabilityPlanningResult:
    """Result returned by MultiCapabilityPlanner."""

    status: MultiCapabilityPlanningStatus
    plan: ExecutionPlan | None = None
    selected_capability_ids: tuple[str, ...] = ()
    error_code: str | None = None
    message: str | None = None

    @property
    def planned(self) -> bool:
        """Return whether a composed ExecutionPlan was produced."""

        return self.status is MultiCapabilityPlanningStatus.PLANNED and self.plan is not None


@dataclass(frozen=True, slots=True)
class _OutputSource:
    variable_name: str
    output_name: str | None = None
    step_id: str | None = None

    def reference(self) -> ExecutionVariableReference:
        if self.output_name is None:
            return ExecutionVariableReference(self.variable_name)
        return ExecutionVariableReference(self.variable_name, (self.output_name,))


@dataclass(frozen=True, slots=True)
class _CapabilityNode:
    capability: CapabilityDefinition
    workflow: WorkflowDefinition


class MultiCapabilityPlanner:
    """Build a composed ExecutionPlan from workflow capability metadata."""

    def __init__(
        self,
        *,
        execution_plan_libraries: Iterable[ExecutionPlanLibrary],
    ) -> None:
        self._libraries = tuple(execution_plan_libraries)
        for library in self._libraries:
            if not isinstance(library, ExecutionPlanLibrary):
                raise MultiCapabilityPlanningError("execution_plan_libraries must contain ExecutionPlanLibrary.")

    def plan(self, request: MultiCapabilityPlanningRequest) -> MultiCapabilityPlanningResult:
        """Plan a deterministic sequence of workflow capabilities."""

        if not isinstance(request, MultiCapabilityPlanningRequest):
            return MultiCapabilityPlanningResult(
                MultiCapabilityPlanningStatus.INVALID_REQUEST,
                error_code="INVALID_REQUEST",
                message="request must be MultiCapabilityPlanningRequest.",
            )

        nodes = self._load_nodes()
        if not nodes:
            return MultiCapabilityPlanningResult(
                MultiCapabilityPlanningStatus.IMPOSSIBLE_DEPENDENCY,
                error_code="NO_CAPABILITIES",
                message="No workflow capabilities are available.",
            )
        required_ids = set(request.required_capability_ids)
        available: dict[str, _OutputSource] = {
            name: _OutputSource(name)
            for name in request.initial_inputs
        }
        used: set[str] = set()
        selected: list[_CapabilityNode] = []
        steps: list[ExecutionStep] = []

        while not self._is_satisfied(request, available, used):
            if len(selected) >= request.maximum_steps:
                return MultiCapabilityPlanningResult(
                    MultiCapabilityPlanningStatus.IMPOSSIBLE_DEPENDENCY,
                    error_code="MAXIMUM_STEPS_EXCEEDED",
                    message="No valid plan was found within the step limit.",
                )

            candidates = self._eligible_nodes(nodes, available, used)
            useful = self._useful_nodes(candidates, nodes, request, available, used)
            if not useful:
                status = (
                    MultiCapabilityPlanningStatus.CYCLE_DETECTED
                    if self._has_cycle(nodes, used)
                    else MultiCapabilityPlanningStatus.IMPOSSIBLE_DEPENDENCY
                )
                return MultiCapabilityPlanningResult(
                    status,
                    error_code=status.value.upper(),
                    message="No deterministic dependency path satisfies the requested outputs.",
                )
            if self._is_ambiguous(useful, nodes, request, available, used):
                return MultiCapabilityPlanningResult(
                    MultiCapabilityPlanningStatus.AMBIGUOUS_GRAPH,
                    error_code="AMBIGUOUS_GRAPH",
                    message="Multiple capability dependency paths are equally valid.",
                )

            node = useful[0]
            step_index = len(selected) + 1
            step_id = f"capability_{step_index}"
            output_variable = f"{step_id}_output"
            dependencies = (steps[-1].id,) if steps else ()
            arguments = {
                input_name: available[input_name].reference()
                for input_name in node.capability.input_names
            }
            step = ExecutionStep(
                step_id,
                f"Run capability {node.capability.capability_id}.",
                None,
                dependencies=dependencies,
                subplan_ref=self._source(node).reference,
                arguments=arguments,
                output_binding=ExecutionVariableBinding(output_variable),
            )
            steps.append(step)
            selected.append(node)
            used.add(node.capability.capability_id)
            for output_name in node.capability.output_names:
                available.setdefault(
                    output_name,
                    _OutputSource(output_variable, output_name=output_name, step_id=step_id),
                )

        output = ExecutionPlanOutput(
            {
                name: available[name].reference()
                for name in request.required_outputs
            }
        )
        plan = ExecutionPlan(
            goal="Execute composed workflow capabilities.",
            ordered_steps=tuple(steps),
            estimated_steps=len(steps),
            required_tools=(),
            detected_risks=(),
            requires_confirmation=False,
            output=output,
        )
        return MultiCapabilityPlanningResult(
            MultiCapabilityPlanningStatus.PLANNED,
            plan=plan,
            selected_capability_ids=tuple(node.capability.capability_id for node in selected),
        )

    def _load_nodes(self) -> tuple[_CapabilityNode, ...]:
        capabilities = WorkflowCapabilityProvider(self._libraries).list_capabilities()
        workflows = {
            (library.library_id, library.version, workflow.reference): workflow
            for library in self._libraries
            for workflow in library.workflows()
        }
        nodes: list[_CapabilityNode] = []
        for capability in capabilities:
            if capability.capability_type is not CapabilityType.WORKFLOW:
                continue
            source = self._source_from_capability(capability)
            workflow = workflows[(source.library.library_id, source.library.library_version, source.reference)]
            nodes.append(_CapabilityNode(capability, workflow))
        return tuple(sorted(nodes, key=lambda node: node.capability.capability_id))

    def _eligible_nodes(
        self,
        nodes: tuple[_CapabilityNode, ...],
        available: dict[str, _OutputSource],
        used: set[str],
    ) -> tuple[_CapabilityNode, ...]:
        available_names = set(available)
        return tuple(
            node
            for node in nodes
            if node.capability.capability_id not in used
            and set(node.capability.input_names).issubset(available_names)
        )

    def _useful_nodes(
        self,
        candidates: tuple[_CapabilityNode, ...],
        nodes: tuple[_CapabilityNode, ...],
        request: MultiCapabilityPlanningRequest,
        available: dict[str, _OutputSource],
        used: set[str],
    ) -> tuple[_CapabilityNode, ...]:
        scored = [
            (self._score(node, nodes, request, available, used), node)
            for node in candidates
        ]
        scored = [(score, node) for score, node in scored if score > 0]
        if not scored:
            return ()
        best = max(score for score, _node in scored)
        return tuple(node for score, node in sorted(scored, key=lambda item: item[1].capability.capability_id) if score == best)

    def _score(
        self,
        node: _CapabilityNode,
        nodes: tuple[_CapabilityNode, ...],
        request: MultiCapabilityPlanningRequest,
        available: dict[str, _OutputSource],
        used: set[str],
    ) -> int:
        outputs = set(node.capability.output_names)
        missing_outputs = set(request.required_outputs) - set(available)
        score = 0
        score += 100 * len(outputs & missing_outputs)
        if node.capability.capability_id in set(request.required_capability_ids) - used:
            score += 80
        unlocks = self._unlocked_inputs(node, nodes, available, used)
        score += 10 * len(unlocks)
        return score

    def _unlocked_inputs(
        self,
        node: _CapabilityNode,
        nodes: tuple[_CapabilityNode, ...],
        available: dict[str, _OutputSource],
        used: set[str],
    ) -> set[str]:
        current = set(available)
        outputs = set(node.capability.output_names)
        unlocked: set[str] = set()
        for other in nodes:
            if other.capability.capability_id in used or other.capability.capability_id == node.capability.capability_id:
                continue
            missing = set(other.capability.input_names) - current
            unlocked.update(missing & outputs)
        return unlocked

    def _is_ambiguous(
        self,
        useful: tuple[_CapabilityNode, ...],
        nodes: tuple[_CapabilityNode, ...],
        request: MultiCapabilityPlanningRequest,
        available: dict[str, _OutputSource],
        used: set[str],
    ) -> bool:
        if len(useful) < 2:
            return False
        missing_outputs = set(request.required_outputs) - set(available)
        direct_sets = [set(node.capability.output_names) & missing_outputs for node in useful]
        if any(outputs for outputs in direct_sets) and len({tuple(sorted(outputs)) for outputs in direct_sets}) == 1:
            return True
        unlock_sets = [
            self._unlocked_inputs(node, nodes, available, used)
            for node in useful
        ]
        return any(unlocks for unlocks in unlock_sets) and len({tuple(sorted(unlocks)) for unlocks in unlock_sets}) == 1

    def _has_cycle(self, nodes: tuple[_CapabilityNode, ...], used: set[str]) -> bool:
        graph: dict[str, set[str]] = {
            node.capability.capability_id: set()
            for node in nodes
            if node.capability.capability_id not in used
        }
        for producer in nodes:
            if producer.capability.capability_id not in graph:
                continue
            outputs = set(producer.capability.output_names)
            for consumer in nodes:
                if consumer.capability.capability_id not in graph or consumer is producer:
                    continue
                if outputs & set(consumer.capability.input_names):
                    graph[producer.capability.capability_id].add(consumer.capability.capability_id)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for dependency in graph[node_id]:
                if visit(dependency):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node_id) for node_id in graph)

    def _is_satisfied(
        self,
        request: MultiCapabilityPlanningRequest,
        available: dict[str, _OutputSource],
        used: set[str],
    ) -> bool:
        return (
            set(request.required_outputs).issubset(set(available))
            and set(request.required_capability_ids).issubset(used)
        )

    def _source(self, node: _CapabilityNode) -> WorkflowCapabilitySource:
        return self._source_from_capability(node.capability)

    def _source_from_capability(self, capability: CapabilityDefinition) -> WorkflowCapabilitySource:
        if not isinstance(capability.source_reference, WorkflowCapabilitySource):
            raise MultiCapabilityPlanningError("workflow capability must use WorkflowCapabilitySource.")
        return capability.source_reference


def _validate_names(values: tuple[str, ...], field_name: str, *, dotted: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidMultiCapabilityPlanningRequestError(f"{field_name} must be iterable.")
    names: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise InvalidMultiCapabilityPlanningRequestError(f"{field_name} values must be non-empty strings.")
        normalized = value.strip().lower() if dotted else value.strip()
        if normalized not in names:
            names.append(normalized)
    return tuple(names)
