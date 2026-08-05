from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time

import pytest

from bootstrap.agent_system import build_core_agent_system
from bootstrap.skill_system import (
    build_builtin_skill_handler_registry,
    register_builtin_skills,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentContextPolicy,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.skill_executor import (
    SkillExecutionRequest,
    SkillExecutionStatus,
    SkillExecutor,
    SkillHandlerRegistry,
)
from core.skill_registry import (
    SkillDefinition,
    SkillExecutionTargetType,
    SkillFieldDefinition,
    SkillLimits,
    SkillRegistry,
)


SKILL_ID = "skill.runtime-limits"
TARGET_ID = "target.runtime-limits"


def _skill(
    *,
    target_type: SkillExecutionTargetType = SkillExecutionTargetType.HANDLER,
    enabled: bool = True,
    input_names: tuple[str, ...] = (),
    output_names: tuple[str, ...] = (),
    input_fields: tuple[SkillFieldDefinition, ...] = (),
    output_fields: tuple[SkillFieldDefinition, ...] = (),
    timeout_seconds: int = 30,
    max_inputs: int = 16,
    max_outputs: int = 16,
    max_result_items: int = 64,
) -> SkillDefinition:
    return SkillDefinition(
        skill_id=SKILL_ID,
        name="Runtime limits",
        version="1.0",
        description="Skill used to verify effective runtime limits.",
        enabled=enabled,
        input_names=input_names,
        output_names=output_names,
        input_fields=input_fields,
        output_fields=output_fields,
        execution_target=TARGET_ID,
        execution_target_type=target_type,
        handler_id=(TARGET_ID if target_type is SkillExecutionTargetType.HANDLER else None),
        limits=SkillLimits(
            timeout_seconds=timeout_seconds,
            max_inputs=max_inputs,
            max_outputs=max_outputs,
            max_result_items=max_result_items,
        ),
    )


class _RecordingExecutor(SkillExecutor):
    def __init__(self, skill: SkillDefinition, output: Mapping[str, object]) -> None:
        super().__init__(skill_registry=SkillRegistry((skill,)))
        self.output = output
        self.calls = 0

    def _execute_target(self, request: SkillExecutionRequest) -> Mapping[str, object]:
        self.calls += 1
        return self.output


@dataclass
class _TargetResult:
    output: Mapping[str, object]

    @property
    def status(self):
        return type("Status", (), {"value": "COMPLETED"})()


class _DelayedTarget:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0

    def _run(self) -> _TargetResult:
        self.calls += 1
        time.sleep(self.delay_seconds)
        return _TargetResult({"result": "ok"})

    def execute(self, *args, **kwargs):
        result = self._run()
        if args and isinstance(args[0], str):
            return result.output
        return result


def _handler_executor(
    skill: SkillDefinition,
    handler,
) -> SkillExecutor:
    handlers = SkillHandlerRegistry()
    handlers.register(TARGET_ID, handler)
    return SkillExecutor(
        skill_registry=SkillRegistry((skill,)),
        handler_registry=handlers,
    )


def _delayed_executor(
    target_type: SkillExecutionTargetType,
    delay_seconds: float,
) -> tuple[SkillDefinition, SkillExecutor, _DelayedTarget | None]:
    skill = _skill(target_type=target_type, timeout_seconds=1)
    registry = SkillRegistry((skill,))
    if target_type is SkillExecutionTargetType.HANDLER:
        calls = {"count": 0}

        def handler(_inputs: Mapping[str, object]) -> Mapping[str, object]:
            calls["count"] += 1
            time.sleep(delay_seconds)
            return {"result": "ok"}

        executor = _handler_executor(skill, handler)
        executor._test_handler_calls = calls  # type: ignore[attr-defined]
        return skill, executor, None
    target = _DelayedTarget(delay_seconds)
    kwargs = {
        "tool_executor": target if target_type is SkillExecutionTargetType.TOOL else None,
        "capability_execution_service": (
            target if target_type is SkillExecutionTargetType.CAPABILITY else None
        ),
        "agent_executor": target if target_type is SkillExecutionTargetType.AGENT else None,
    }
    return skill, SkillExecutor(skill_registry=registry, **kwargs), target


def _denied_agent() -> AgentDefinition:
    return AgentDefinition(
        agent_id="agent.runtime-limits",
        agent_type=AgentType.GENERAL,
        name="Runtime limits agent",
        description="Agent denied access to the runtime limits skill.",
        capabilities=AgentCapabilities(),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
        metadata={"denied_skill_ids": SKILL_ID},
    )


def test_max_inputs_exceeded_does_not_invoke_target() -> None:
    skill = _skill(max_inputs=1)
    executor = _RecordingExecutor(skill, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill, inputs={"first": 1, "second": 2}))

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "SKILL_INPUT_LIMIT_EXCEEDED"
    assert executor.calls == 0


def test_max_inputs_within_limit_executes_normally() -> None:
    skill = _skill(max_inputs=1)
    executor = _RecordingExecutor(skill, {"result": "ok"})

    result = executor.execute(SkillExecutionRequest(skill, inputs={"first": 1}))

    assert result.status is SkillExecutionStatus.COMPLETED
    assert executor.calls == 1


def test_max_outputs_exceeded_rejects_executed_result() -> None:
    skill = _skill(max_outputs=1)
    executor = _RecordingExecutor(skill, {"first": 1, "second": 2})

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "SKILL_OUTPUT_LIMIT_EXCEEDED"
    assert executor.calls == 1


def test_max_result_items_rejects_large_nested_list() -> None:
    skill = _skill(max_result_items=2)
    executor = _RecordingExecutor(skill, {"result": {"items": [1, 2, 3]}})

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "SKILL_RESULT_ITEM_LIMIT_EXCEEDED"


def test_max_result_items_accepts_list_and_tuple_within_limit() -> None:
    skill = _skill(max_result_items=2)
    executor = _RecordingExecutor(skill, {"first": [1, 2], "second": (3, 4)})

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.COMPLETED


@pytest.mark.parametrize("target_type", tuple(SkillExecutionTargetType))
def test_timeout_recovers_control_for_every_target(
    target_type: SkillExecutionTargetType,
) -> None:
    skill, executor, target = _delayed_executor(target_type, 2.0)

    started = time.monotonic()
    result = executor.execute(SkillExecutionRequest(skill))
    elapsed = time.monotonic() - started

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "SKILL_EXECUTION_TIMEOUT"
    assert elapsed < 1.75
    if target is not None:
        assert target.calls == 1


def test_handler_within_timeout_completes() -> None:
    skill, executor, _target = _delayed_executor(SkillExecutionTargetType.HANDLER, 0.01)

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.COMPLETED


def test_skill_with_default_unconfigured_limits_preserves_execution() -> None:
    skill = SkillDefinition(
        skill_id=SKILL_ID,
        name="Default limits",
        version="1.0",
        description="Skill with no explicit limits configuration.",
        execution_target=TARGET_ID,
        execution_target_type=SkillExecutionTargetType.HANDLER,
        handler_id=TARGET_ID,
    )
    executor = _handler_executor(skill, lambda _inputs: {"result": "ok"})

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.COMPLETED


def test_input_contract_violation_precedes_input_limit() -> None:
    skill = _skill(input_names=("text",), max_inputs=1)
    executor = _RecordingExecutor(skill, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill, inputs={"first": 1, "second": 2}))

    assert result.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert executor.calls == 0


def test_typed_input_contract_violation_precedes_input_limit() -> None:
    skill = _skill(
        input_fields=(SkillFieldDefinition("text", "string"),),
        max_inputs=1,
    )
    executor = _RecordingExecutor(skill, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill, inputs={"text": 1, "extra": 2}))

    assert result.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert executor.calls == 0


def test_output_contract_violation_precedes_output_limit() -> None:
    skill = _skill(output_names=("result",), max_outputs=1)
    executor = _RecordingExecutor(skill, {"first": 1, "second": 2})

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.error_code == "SKILL_OUTPUT_CONTRACT_VIOLATION"
    assert executor.calls == 1


def test_real_target_exception_keeps_existing_error_code() -> None:
    skill = _skill()

    def fail(_inputs: Mapping[str, object]) -> Mapping[str, object]:
        raise RuntimeError("controlled target failure")

    result = _handler_executor(skill, fail).execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "RuntimeError"


@pytest.mark.parametrize(
    ("text", "expected"),
    (("Atlas", "ATLAS"), ("hola atlas", "HOLA ATLAS")),
)
def test_builtin_text_uppercase_remains_end_to_end(text: str, expected: str) -> None:
    built = build_core_agent_system(
        skill_handler_registry=build_builtin_skill_handler_registry(),
    )
    assert built.system is not None
    skill_system = built.system.skill_system
    register_builtin_skills(skill_system)
    skill = skill_system.skill_registry.get("skill.text-uppercase")

    result = skill_system.skill_executor.execute(
        SkillExecutionRequest(skill, inputs={"text": text})
    )

    assert result.status is SkillExecutionStatus.COMPLETED
    assert result.output == {"result": expected}


def test_unregistered_skill_is_rejected_without_execution() -> None:
    skill = _skill()
    executor = SkillExecutor(skill_registry=SkillRegistry())

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.TARGET_UNAVAILABLE
    assert result.error_code == "SKILL_NOT_REGISTERED"


def test_disabled_skill_is_rejected_without_execution() -> None:
    skill = _skill(enabled=False)
    executor = _RecordingExecutor(skill, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill))

    assert result.status is SkillExecutionStatus.SKILL_DISABLED
    assert result.error_code == "SKILL_DISABLED"
    assert executor.calls == 0


def test_authorization_denied_is_rejected_without_execution() -> None:
    skill = _skill()
    executor = _RecordingExecutor(skill, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill, agent=_denied_agent()))

    assert result.status is SkillExecutionStatus.SKILL_NOT_AUTHORIZED
    assert result.error_code == "SKILL_NOT_AUTHORIZED"
    assert executor.calls == 0
