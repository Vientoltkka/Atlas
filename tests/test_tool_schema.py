from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from core.execution_plan_executor import (
    ExecutionControl,
    ExecutionErrorCode,
    ExecutionPlanExecutor,
    ResumableExecutionState,
)
from core.execution_context import ExecutionContext
from core.execution_variable_reference import ExecutionVariableReference
from core.execution_retry import RetryPolicy
from core.execution_plan_validator import ExecutionPlanValidator, plan_signature
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference
from tools.base_tool import BaseTool
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext
from tools.tool_schema import (
    ToolArgumentsSchema,
    ToolParameterSchema,
    ToolSchemaErrorCode,
    ToolSchemaValidationException,
    ToolSchemaValidationResult,
)


class CapturingTool(BaseTool):
    def __init__(
        self,
        name: str = "demo.tool",
        output: Any = "ok",
    ) -> None:
        self._name = name
        self._output = output
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Tool {self._name}."

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self.contexts.append(context)
        return self._output


class SequenceTool(CapturingTool):
    def __init__(
        self,
        outputs: list[Any],
        name: str = "demo.tool",
    ) -> None:
        super().__init__(name)
        self._outputs = list(outputs)

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self.contexts.append(context)
        return self._outputs.pop(0)


def _schema(
    *parameters: ToolParameterSchema,
    allow_extra_arguments: bool = False,
) -> ToolArgumentsSchema:
    return ToolArgumentsSchema(
        parameters=parameters,
        allow_extra_arguments=allow_extra_arguments,
    )


def _registry(
    tool: CapturingTool,
    schema: ToolArgumentsSchema | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool, arguments_schema=schema)
    return registry


def _step(arguments: dict[str, Any] | None = None) -> ExecutionStep:
    return ExecutionStep(
        id="step_1",
        description="Run tool.",
        tool="demo.tool",
        arguments={} if arguments is None else arguments,
    )


def _plan(arguments: dict[str, Any] | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Run demo tool.",
        ordered_steps=(_step(arguments),),
        estimated_steps=1,
        required_tools=("demo.tool",),
        detected_risks=(),
        requires_confirmation=False,
    )


def _codes(result: ToolSchemaValidationResult) -> tuple[str, ...]:
    return tuple(error.error_code for error in result.errors)


def test_creates_valid_tool_parameter_schema() -> None:
    parameter = ToolParameterSchema(
        "query",
        str,
        required=True,
        description="Search query.",
    )

    assert parameter.name == "query"
    assert parameter.value_type is str
    assert parameter.required is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "", "value_type": str}, "name cannot be empty"),
        ({"name": "when", "value_type": tuple}, "Unsupported"),
        ({"name": "limit", "value_type": int, "minimum": 3, "maximum": 2}, "minimum"),
        ({"name": "text", "value_type": str, "minimum": 1}, "minimum"),
        ({"name": "flag", "value_type": bool, "maximum": 1}, "maximum"),
        ({"name": "count", "value_type": int, "default": "1"}, "default"),
        ({"name": "order", "value_type": str, "choices": ("asc", 1)}, "choices"),
    ],
)
def test_invalid_parameter_schema_is_rejected(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ToolParameterSchema(**kwargs)


def test_optional_required_allow_none_and_defaults() -> None:
    schema = _schema(
        ToolParameterSchema("required", str, required=True),
        ToolParameterSchema("optional", int),
        ToolParameterSchema("nullable", str, allow_none=True),
        ToolParameterSchema("limit", int, default=10),
    )

    result = schema.validate(
        "demo.tool",
        {"required": "x", "nullable": None},
    )

    assert result.is_valid is True
    assert dict(result.normalized_arguments) == {
        "required": "x",
        "nullable": None,
        "limit": 10,
    }


def test_mutable_default_and_choices_are_protected() -> None:
    default_items = ["a"]
    choices = ["asc", "desc"]
    parameter = ToolParameterSchema("items", list, default=default_items)
    order = ToolParameterSchema("order", str, choices=choices)
    default_items.append("b")
    choices.append("random")

    first = _schema(parameter).validate("demo.tool", {})
    second = _schema(parameter).validate("demo.tool", {})
    first.normalized_arguments["items"].append("changed")  # type: ignore[index,union-attr]

    assert dict(second.normalized_arguments) == {"items": ["a"]}
    assert order.choices == ("asc", "desc")


def test_required_optional_unknown_and_structured_result() -> None:
    schema = _schema(ToolParameterSchema("query", str, required=True))

    result = schema.validate("demo.tool", {"extra": "x"})

    assert isinstance(result, ToolSchemaValidationResult)
    assert result.is_valid is False
    assert _codes(result) == (
        ToolSchemaErrorCode.UNKNOWN_PARAMETER.value,
        ToolSchemaErrorCode.REQUIRED_PARAMETER_MISSING.value,
    )
    assert result.errors[1].tool_name == "demo.tool"
    assert result.errors[1].parameter_name == "query"
    assert "required parameter missing" in result.errors[1].message


def test_unknown_argument_can_be_allowed_and_preserved() -> None:
    schema = _schema(
        ToolParameterSchema("query", str, required=True),
        allow_extra_arguments=True,
    )

    result = schema.validate("demo.tool", {"query": "x", "extra": {"a": 1}})

    assert result.is_valid is True
    assert dict(result.normalized_arguments) == {"query": "x", "extra": {"a": 1}}


@pytest.mark.parametrize(
    ("parameter", "value", "valid"),
    [
        (ToolParameterSchema("enabled", bool), True, True),
        (ToolParameterSchema("count", int), 1, True),
        (ToolParameterSchema("ratio", float), 1.5, True),
        (ToolParameterSchema("text", str), "x", True),
        (ToolParameterSchema("items", list), ["x"], True),
        (ToolParameterSchema("payload", dict), {"x": 1}, True),
        (ToolParameterSchema("count", int), True, False),
        (ToolParameterSchema("ratio", float), True, False),
        (ToolParameterSchema("count", int), "20", False),
        (ToolParameterSchema("text", str), 20, False),
        (ToolParameterSchema("text", str), None, False),
        (ToolParameterSchema("text", str, allow_none=True), None, True),
    ],
)
def test_type_and_none_validation(
    parameter: ToolParameterSchema,
    value: object,
    valid: bool,
) -> None:
    result = _schema(parameter).validate("demo.tool", {parameter.name: value})

    assert result.is_valid is valid


def test_choices_and_numeric_limits_are_enforced() -> None:
    schema = _schema(
        ToolParameterSchema("order", str, choices=("asc", "desc")),
        ToolParameterSchema("minimum", int, minimum=1),
        ToolParameterSchema("maximum", int, maximum=10),
        ToolParameterSchema("both", float, minimum=1.5, maximum=3.5),
    )

    valid = schema.validate(
        "demo.tool",
        {"order": "asc", "minimum": 1, "maximum": 10, "both": 3.5},
    )
    invalid = schema.validate(
        "demo.tool",
        {"order": "ASC", "minimum": 0, "maximum": 11, "both": 4.0},
    )

    assert valid.is_valid is True
    assert _codes(invalid) == (
        ToolSchemaErrorCode.INVALID_CHOICE.value,
        ToolSchemaErrorCode.BELOW_MINIMUM.value,
        ToolSchemaErrorCode.ABOVE_MAXIMUM.value,
        ToolSchemaErrorCode.ABOVE_MAXIMUM.value,
    )


def test_rejects_non_finite_float_values() -> None:
    result = _schema(ToolParameterSchema("ratio", float)).validate(
        "demo.tool",
        {"ratio": float("inf")},
    )

    assert result.is_valid is False
    assert _codes(result) == (ToolSchemaErrorCode.INVALID_TYPE.value,)


def test_normalized_arguments_are_independent_from_source() -> None:
    source = {"items": ["a"], "nested": {"value": 1}}
    result = _schema(
        ToolParameterSchema("items", list),
        ToolParameterSchema("nested", dict),
    ).validate("demo.tool", source)
    source["items"].append("b")  # type: ignore[union-attr]
    source["nested"]["value"] = 2  # type: ignore[index]

    assert dict(result.normalized_arguments) == {
        "items": ["a"],
        "nested": {"value": 1},
    }


def test_tool_registry_registers_and_recovers_schema() -> None:
    tool = CapturingTool()
    schema = _schema(ToolParameterSchema("query", str))
    registry = _registry(tool, schema)
    no_schema = CapturingTool("legacy.tool")
    registry.register(no_schema)

    assert registry.arguments_schema("demo.tool") is schema
    assert registry.descriptor("demo.tool").arguments_schema is schema
    assert registry.arguments_schema("legacy.tool") is None


def test_plan_validator_validates_step_arguments_against_registry_schema() -> None:
    registry = _registry(CapturingTool(), _schema(ToolParameterSchema("query", str, required=True)))

    valid = ExecutionPlanValidator(registry).validate(_plan({"query": "x"}))
    missing = ExecutionPlanValidator(registry).validate(_plan({}))
    wrong_type = ExecutionPlanValidator(registry).validate(_plan({"query": 1}))
    unknown = ExecutionPlanValidator(registry).validate(_plan({"query": "x", "extra": True}))

    assert valid.is_valid is True
    assert missing.is_valid is False
    assert "required parameter missing" in missing.errors[0]
    assert wrong_type.is_valid is False
    assert "expected str, got int" in wrong_type.errors[0]
    assert unknown.is_valid is False
    assert "unknown parameter" in unknown.errors[0]


def test_tool_executor_validates_applies_defaults_and_keeps_context_independent() -> None:
    tool = CapturingTool()
    registry = _registry(
        tool,
        _schema(
            ToolParameterSchema("query", str, required=True),
            ToolParameterSchema("max_results", int, default=10),
        ),
    )
    original = {"query": "is:unread"}

    result = ToolExecutor(registry).execute("demo.tool", arguments=original)
    original["query"] = "changed"

    assert result == "ok"
    assert tool.contexts[0].parameters == {"query": "is:unread", "max_results": 10}


def test_tool_executor_rejects_invalid_arguments_before_tool_call() -> None:
    tool = CapturingTool()
    registry = _registry(tool, _schema(ToolParameterSchema("query", str, required=True)))

    with pytest.raises(ToolSchemaValidationException) as raised:
        ToolExecutor(registry).execute("demo.tool", arguments={"query": 1})

    assert raised.value.result.errors[0].error_code == ToolSchemaErrorCode.INVALID_TYPE.value
    assert tool.contexts == []


def test_tool_executor_rejects_unresolved_execution_variable_reference() -> None:
    tool = CapturingTool()
    registry = _registry(tool)

    with pytest.raises(ValueError):
        ToolExecutor(registry).execute(
            "demo.tool",
            arguments={"query": ExecutionVariableReference("workspace_path")},
        )

    assert tool.contexts == []


def test_tool_without_schema_keeps_working_and_steps_without_arguments_work() -> None:
    tool = CapturingTool()
    registry = _registry(tool)
    plan = _plan()

    direct = ToolExecutor(registry).execute("demo.tool", arguments={"anything": "goes"})
    planned = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert direct == "ok"
    assert planned.success is True
    assert tool.contexts[-1].parameters == {}


def test_execution_plan_executor_uses_normalized_arguments_for_retries() -> None:
    tool = SequenceTool(
        [
            {"success": False, "error_code": "TRANSIENT_ERROR", "error": "again"},
            "done",
        ],
    )
    registry = _registry(tool, _schema(ToolParameterSchema("query", str, default="all")))
    plan = _plan({})
    result = ExecutionPlanExecutor(
        registry,
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.success is True
    assert tool.contexts[0].parameters == {"query": "all"}
    assert tool.contexts[1].parameters == {"query": "all"}
    assert dict(plan.ordered_steps[0].arguments) == {}


def test_execution_plan_executor_validates_schema_after_variable_resolution() -> None:
    tool = CapturingTool()
    registry = _registry(tool, _schema(ToolParameterSchema("query", str, required=True)))
    plan = _plan({"query": ExecutionVariableReference("search_query")})
    context = ExecutionContext("exec-schema-1", initial_variables={"search_query": "atlas"})

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
        execution_context=context,
    )

    assert result.success is True
    assert tool.contexts[0].parameters == {"query": "atlas"}


def test_execution_plan_executor_fails_schema_after_wrong_variable_type() -> None:
    tool = CapturingTool()
    registry = _registry(tool, _schema(ToolParameterSchema("query", str, required=True)))
    plan = _plan({"query": ExecutionVariableReference("search_query")})
    context = ExecutionContext("exec-schema-1", initial_variables={"search_query": 10})

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
        execution_context=context,
    )

    assert result.success is False
    assert result.error_code == ExecutionErrorCode.TOOL_SCHEMA_VALIDATION_FAILED.value
    assert tool.contexts == []


def test_resume_revalidates_current_schema_and_schema_change_can_invalidate_resume() -> None:
    tool = CapturingTool(output="done")
    old_registry = _registry(tool, _schema(ToolParameterSchema("query", str, required=True)))
    plan = _plan({"query": "x"})
    validation = ExecutionPlanValidator(old_registry).validate(plan)
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=(),
        pending_step_ids=("step_1",),
        failed_step_ids=(),
        interrupted_step_id="step_1",
        previous_results={},
        resumable=True,
    )
    new_registry = _registry(CapturingTool(), _schema(ToolParameterSchema("query", int, required=True)))

    resumed = ExecutionPlanExecutor(new_registry).resume(state)

    assert resumed.success is False
    assert resumed.error_code == ExecutionErrorCode.TOOL_SCHEMA_VALIDATION_FAILED.value
    assert "Current tool schema is incompatible" in (resumed.error or "")


def test_plan_signature_does_not_depend_on_schema_configuration() -> None:
    plan = _plan({"query": 10})
    first_registry = _registry(
        CapturingTool(),
        _schema(ToolParameterSchema("query", int, minimum=1, maximum=100)),
    )
    second_registry = _registry(
        CapturingTool(),
        _schema(ToolParameterSchema("query", int, minimum=1, maximum=200)),
    )

    first = ExecutionPlanValidator(first_registry).validate(plan)
    second = ExecutionPlanValidator(second_registry).validate(plan)

    assert first.is_valid is True
    assert second.is_valid is True
    assert first.plan_signature == second.plan_signature == plan_signature(plan)


def test_observability_records_schema_validation_without_values() -> None:
    secret = "secret-value"
    tool = CapturingTool()
    registry = _registry(tool, _schema(ToolParameterSchema("query", str, required=True)))
    plan = _plan({"query": secret})

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    actions = [event.action for event in result.trace.events]  # type: ignore[union-attr]
    details = [event.details for event in result.trace.events]  # type: ignore[union-attr]
    assert "schema_validation_started" in actions
    assert "schema_validation_succeeded" in actions
    assert secret not in repr(details)


def test_observability_records_schema_validation_failure_without_values() -> None:
    secret = "secret-value"
    first = CapturingTool("first.tool", {"count": secret})
    second = CapturingTool("second.tool")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second, arguments_schema=_schema(ToolParameterSchema("count", int, required=True)))
    plan = ExecutionPlan(
        goal="Trace schema validation failure.",
        ordered_steps=(
            ExecutionStep("step_1", "First.", "first.tool"),
            ExecutionStep(
                "step_2",
                "Second.",
                "second.tool",
                dependencies=("step_1",),
                arguments={"count": {"$ref": "steps.step_1.output.count"}},
            ),
        ),
        estimated_steps=2,
        required_tools=("first.tool", "second.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )
    validation = ExecutionPlanValidator(registry).validate(plan)

    result = ExecutionPlanExecutor(registry).execute(plan, validation)

    assert result.error_code == ExecutionErrorCode.TOOL_SCHEMA_VALIDATION_FAILED.value
    assert result.trace is not None
    failed_events = [
        event
        for event in result.trace.events
        if event.action == "schema_validation_failed"
    ]
    assert failed_events
    assert failed_events[0].details["error_count"] == 1
    assert failed_events[0].details["invalid_parameters"] == ["count"]
    assert secret not in repr(failed_events[0].details)


def test_no_string_to_int_coercion_and_execution_arguments_are_not_mutated() -> None:
    tool = CapturingTool()
    registry = _registry(tool, _schema(ToolParameterSchema("count", int, default=1)))
    plan = _plan({"count": "20"})
    before = plan.ordered_steps[0].arguments.as_dict()
    validation = replace(ExecutionPlanValidator().validate(plan), plan_signature=plan_signature(plan))

    result = ExecutionPlanExecutor(registry).execute(plan, validation)

    assert result.success is False
    assert tool.contexts == []
    assert plan.ordered_steps[0].arguments.as_dict() == before


def test_execution_can_be_interrupted_and_resumed_after_schema_validation() -> None:
    first = CapturingTool("first.tool", {"content": "alpha"})
    second = CapturingTool("second.tool")
    registry = ToolRegistry()
    registry.register(first, arguments_schema=_schema(ToolParameterSchema("query", str, default="all")))
    registry.register(second, arguments_schema=_schema(ToolParameterSchema("content", str, required=True)))
    plan = ExecutionPlan(
        goal="Resume with schemas.",
        ordered_steps=(
            ExecutionStep("step_1", "First.", "first.tool", arguments={}),
            ExecutionStep(
                "step_2",
                "Second.",
                "second.tool",
                dependencies=("step_1",),
                arguments={"content": {"$ref": "steps.step_1.output.content"}},
            ),
        ),
        estimated_steps=2,
        required_tools=("first.tool", "second.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )
    validation = ExecutionPlanValidator(registry).validate(plan)
    interrupted = ExecutionPlanExecutor(registry).execute(
        plan,
        validation,
        control=ExecutionControl(should_stop=lambda: bool(first.contexts)),
    )
    state = ResumableExecutionState(
        objective="resume",
        original_plan=plan,
        validation_result=validation,
        validated_plan_signature=validation.plan_signature,
        completed_step_ids=tuple(interrupted.completed_steps),
        pending_step_ids=tuple(interrupted.pending_steps),
        failed_step_ids=tuple(interrupted.failed_steps),
        interrupted_step_id=interrupted.current_step,
        previous_results={
            result.step_id: result.output
            for result in interrupted.step_results
            if result.success
        },
        resumable=interrupted.resumable,
    )

    resumed = ExecutionPlanExecutor(registry).resume(state)

    assert resumed.success is True
    assert first.contexts[0].parameters == {"query": "all"}
    assert second.contexts[0].parameters == {"content": "alpha"}


def test_structured_reference_is_validated_against_schema_after_resolution() -> None:
    first = CapturingTool("read.tool", "content")
    second = CapturingTool("consume.tool")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second, arguments_schema=_schema(ToolParameterSchema("text", str, required=True)))
    plan = ExecutionPlan(
        goal="Resolve then validate.",
        ordered_steps=(
            ExecutionStep("read", "Read.", "read.tool"),
                ExecutionStep(
                    "consume",
                    "Consume.",
                    "consume.tool",
                    depends_on=("read",),
                    arguments={"text": StepOutputReference("read")},
                ),
        ),
        estimated_steps=2,
        required_tools=("read.tool", "consume.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.success is True
    assert second.contexts[0].parameters == {"text": "content"}


def test_structured_reference_wrong_resolved_type_fails_schema_validation() -> None:
    first = CapturingTool("read.tool", {"content": "not text"})
    second = CapturingTool("consume.tool")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second, arguments_schema=_schema(ToolParameterSchema("text", str, required=True)))
    plan = ExecutionPlan(
        goal="Resolve wrong type.",
        ordered_steps=(
            ExecutionStep("read", "Read.", "read.tool"),
                ExecutionStep(
                    "consume",
                    "Consume.",
                    "consume.tool",
                    depends_on=("read",),
                    arguments={"text": StepOutputReference("read")},
                ),
        ),
        estimated_steps=2,
        required_tools=("read.tool", "consume.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.success is False
    assert result.error_code == ExecutionErrorCode.TOOL_SCHEMA_VALIDATION_FAILED.value
    assert second.contexts == []


def test_schema_default_and_unknown_argument_policy_apply_after_structured_resolution() -> None:
    first = CapturingTool("read.tool", "query")
    second = CapturingTool("consume.tool")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(
        second,
        arguments_schema=_schema(
            ToolParameterSchema("query", str, required=True),
            ToolParameterSchema("limit", int, default=5),
        ),
    )
    plan = ExecutionPlan(
        goal="Resolve with defaults.",
        ordered_steps=(
            ExecutionStep("read", "Read.", "read.tool"),
                ExecutionStep(
                    "consume",
                    "Consume.",
                    "consume.tool",
                    depends_on=("read",),
                    arguments={"query": StepOutputReference("read")},
                ),
        ),
        estimated_steps=2,
        required_tools=("read.tool", "consume.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.success is True
    assert second.contexts[0].parameters == {"query": "query", "limit": 5}


def test_retry_resolves_structured_reference_from_original_arguments_each_attempt() -> None:
    first = CapturingTool("read.tool", {"content": "alpha"})
    second = SequenceTool(
        [
            {"success": False, "error_code": "TRANSIENT_ERROR", "error": "again"},
            "done",
        ],
        name="consume.tool",
    )
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second, arguments_schema=_schema(ToolParameterSchema("content", str, required=True)))
    reference = StepOutputReference("read", ("content",))
    plan = ExecutionPlan(
        goal="Retry structured reference.",
        ordered_steps=(
            ExecutionStep("read", "Read.", "read.tool"),
            ExecutionStep(
                "consume",
                "Consume.",
                "consume.tool",
                dependencies=("read",),
                arguments={"content": reference},
            ),
        ),
        estimated_steps=2,
        required_tools=("read.tool", "consume.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = ExecutionPlanExecutor(
        registry,
        retry_policy=RetryPolicy(max_attempts=2),
    ).execute(plan, ExecutionPlanValidator(registry).validate(plan))

    assert result.success is True
    assert second.contexts[0].parameters == {"content": "alpha"}
    assert second.contexts[1].parameters == {"content": "alpha"}
    assert plan.ordered_steps[1].arguments.as_dict() == {"content": reference}


def test_parameter_resolution_observability_does_not_record_values() -> None:
    secret = "secret-value"
    first = CapturingTool("read.tool", {"content": secret})
    second = CapturingTool("consume.tool")
    registry = ToolRegistry()
    registry.register(first)
    registry.register(second, arguments_schema=_schema(ToolParameterSchema("content", str, required=True)))
    plan = ExecutionPlan(
        goal="Observe parameter resolution.",
        ordered_steps=(
            ExecutionStep("read", "Read.", "read.tool"),
            ExecutionStep(
                "consume",
                "Consume.",
                "consume.tool",
                dependencies=("read",),
                arguments={"content": StepOutputReference("read", ("content",))},
            ),
        ),
        estimated_steps=2,
        required_tools=("read.tool", "consume.tool"),
        detected_risks=(),
        requires_confirmation=False,
    )

    result = ExecutionPlanExecutor(registry).execute(
        plan,
        ExecutionPlanValidator(registry).validate(plan),
    )

    assert result.trace is not None
    actions = [event.action for event in result.trace.events]
    details = [event.details for event in result.trace.events]
    assert "parameter_resolution_started" in actions
    assert "parameter_resolution_succeeded" in actions
    assert secret not in repr(details)
