from __future__ import annotations

from collections.abc import Mapping

import pytest

from bootstrap.agent_system import build_core_agent_system
from bootstrap.skill_system import (
    build_builtin_skill_handler_registry,
    register_builtin_skills,
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
    SkillRegistry,
)


SKILL_ID = "skill.contract-test"


def _skill(
    *,
    target_type: SkillExecutionTargetType = SkillExecutionTargetType.HANDLER,
    input_names: tuple[str, ...] = ("text",),
    output_names: tuple[str, ...] = ("result",),
) -> SkillDefinition:
    return SkillDefinition(
        skill_id=SKILL_ID,
        name="Contract test",
        version="1.0",
        description="Skill used to verify central contract validation.",
        input_names=input_names,
        output_names=output_names,
        execution_target="handler.contract-test",
        execution_target_type=target_type,
        handler_id=(
            "handler.contract-test"
            if target_type is SkillExecutionTargetType.HANDLER
            else None
        ),
    )


class _RecordingExecutor(SkillExecutor):
    def __init__(
        self,
        skill_registry: SkillRegistry,
        output: Mapping[str, object],
    ) -> None:
        super().__init__(skill_registry=skill_registry)
        self.output = output
        self.calls = 0

    def _execute_target(self, request: SkillExecutionRequest) -> Mapping[str, object]:
        self.calls += 1
        return self.output


@pytest.mark.parametrize("target_type", tuple(SkillExecutionTargetType))
def test_missing_required_input_is_rejected_before_every_target(
    target_type: SkillExecutionTargetType,
) -> None:
    skill = _skill(target_type=target_type)
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(registry, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill, inputs={}))

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert result.output is None
    assert executor.calls == 0


@pytest.mark.parametrize("target_type", tuple(SkillExecutionTargetType))
@pytest.mark.parametrize(
    ("output_names", "output"),
    (
        (("result",), {}),
        (("result", "metadata"), {"result": "ATLAS"}),
    ),
)
def test_incomplete_output_is_not_reported_as_completed_for_any_target(
    target_type: SkillExecutionTargetType,
    output_names: tuple[str, ...],
    output: Mapping[str, object],
) -> None:
    skill = _skill(target_type=target_type, output_names=output_names)
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(registry, output)

    result = executor.execute(
        SkillExecutionRequest(skill, inputs={"text": "Atlas"})
    )

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "SKILL_OUTPUT_CONTRACT_VIOLATION"
    assert result.output is None
    assert executor.calls == 1


def test_contract_validation_uses_the_authoritative_registered_definition() -> None:
    registered = _skill()
    caller_definition = _skill(input_names=(), output_names=())
    registry = SkillRegistry((registered,))
    executor = _RecordingExecutor(registry, {})

    result = executor.execute(SkillExecutionRequest(caller_definition, inputs={}))

    assert result.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert executor.calls == 0


def test_real_handler_failure_is_not_reclassified_as_output_violation() -> None:
    skill = _skill()
    registry = SkillRegistry((skill,))
    handlers = SkillHandlerRegistry()

    def fail(_inputs: Mapping[str, object]) -> Mapping[str, object]:
        raise RuntimeError("controlled handler failure")

    handlers.register("handler.contract-test", fail)
    executor = SkillExecutor(skill_registry=registry, handler_registry=handlers)

    result = executor.execute(
        SkillExecutionRequest(skill, inputs={"text": "Atlas"})
    )

    assert result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert result.error_code == "RuntimeError"
    assert result.output is None


def test_additional_inputs_and_outputs_remain_compatible() -> None:
    skill = _skill()
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(
        registry,
        {"result": "ATLAS", "additional_output": True},
    )

    result = executor.execute(
        SkillExecutionRequest(
            skill,
            inputs={"text": "Atlas", "additional_input": True},
        )
    )

    assert result.status is SkillExecutionStatus.COMPLETED
    assert result.output == {"additional_output": True, "result": "ATLAS"}


@pytest.mark.parametrize(
    ("text", "expected"),
    (("Atlas", "ATLAS"), ("hola atlas", "HOLA ATLAS")),
)
def test_builtin_text_uppercase_remains_valid(text: str, expected: str) -> None:
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
