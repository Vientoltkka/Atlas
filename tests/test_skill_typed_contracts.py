from __future__ import annotations

from collections.abc import Mapping
import json

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
)
from core.skill_manifest import SkillManifestLoader, SkillManifestStatus
from core.skill_registry import (
    SkillDefinition,
    SkillExecutionTargetType,
    SkillRegistry,
)
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus


SKILL_ID = "skill.typed-contract-test"


def _manifest(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "1.0",
        "skill_id": SKILL_ID,
        "name": "Typed contract test",
        "version": "1.0",
        "description": "Skill used to verify typed contracts.",
        "enabled": True,
        "input_fields": [
            {"name": "value", "type": "string", "required": True},
        ],
        "output_fields": [
            {"name": "result", "type": "string", "required": True},
        ],
        "execution_target": "handler.typed-contract-test",
        "execution_target_type": "handler",
        "handler_id": "handler.typed-contract-test",
    }
    data.update(overrides)
    return data


def _load_definition(**overrides: object) -> SkillDefinition:
    loaded = SkillManifestLoader().load(json.dumps(_manifest(**overrides)))
    assert loaded.status is SkillManifestStatus.VALID, loaded.errors
    assert loaded.definition is not None
    return loaded.definition


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


def _execute(
    skill: SkillDefinition,
    value: object,
    *,
    output: Mapping[str, object] | None = None,
) -> tuple[object, _RecordingExecutor]:
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(
        registry,
        {"result": "ok"} if output is None else output,
    )
    result = executor.execute(
        SkillExecutionRequest(skill, inputs={"value": value})
    )
    return result, executor


def test_typed_manifest_normalizes_immutable_field_definitions() -> None:
    skill = _load_definition(
        input_fields=[
            {"name": "value", "type": "string", "required": True},
            {"name": "language", "type": "string", "required": False},
        ]
    )

    assert skill.input_names == ()
    assert skill.output_names == ()
    assert [
        (field.name, field.type_name, field.required)
        for field in skill.input_fields
    ] == [("value", "string", True), ("language", "string", False)]
    assert [
        (field.name, field.type_name, field.required)
        for field in skill.output_fields
    ] == [("result", "string", True)]


@pytest.mark.parametrize(
    ("type_name", "value", "completed"),
    (
        ("string", "Atlas", True),
        ("string", 42, False),
        ("integer", 42, True),
        ("integer", True, False),
        ("number", 42, True),
        ("number", 4.2, True),
        ("number", False, False),
        ("boolean", True, True),
        ("object", {"name": "Atlas"}, True),
        ("array", ["Atlas"], True),
    ),
)
def test_input_type_contracts(
    type_name: str,
    value: object,
    completed: bool,
) -> None:
    skill = _load_definition(
        input_fields=[
            {"name": "value", "type": type_name, "required": True},
        ]
    )

    result, executor = _execute(skill, value)

    assert result.completed is completed
    assert result.error_code == (
        None if completed else "SKILL_INPUT_CONTRACT_VIOLATION"
    )
    assert executor.calls == (1 if completed else 0)


def test_required_input_missing_is_rejected_without_execution() -> None:
    skill = _load_definition()
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(registry, {"result": "unused"})

    result = executor.execute(SkillExecutionRequest(skill, inputs={}))

    assert result.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert executor.calls == 0


@pytest.mark.parametrize(
    ("inputs", "completed"),
    (
        ({"value": "Atlas"}, True),
        ({"value": "Atlas", "language": "es"}, True),
        ({"value": "Atlas", "language": 1}, False),
    ),
)
def test_optional_input_may_be_absent_but_is_typed_when_present(
    inputs: Mapping[str, object],
    completed: bool,
) -> None:
    skill = _load_definition(
        input_fields=[
            {"name": "value", "type": "string", "required": True},
            {"name": "language", "type": "string", "required": False},
        ]
    )
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(registry, {"result": "ok"})

    result = executor.execute(SkillExecutionRequest(skill, inputs=inputs))

    assert result.completed is completed
    assert executor.calls == (1 if completed else 0)


@pytest.mark.parametrize(
    ("output_fields", "output", "completed"),
    (
        (
            [{"name": "result", "type": "string", "required": True}],
            {},
            False,
        ),
        (
            [{"name": "result", "type": "string", "required": True}],
            {"result": 42},
            False,
        ),
        (
            [
                {"name": "result", "type": "string", "required": True},
                {"name": "language", "type": "string", "required": False},
            ],
            {"result": "ATLAS"},
            True,
        ),
        (
            [
                {"name": "result", "type": "string", "required": True},
                {"name": "language", "type": "string", "required": False},
            ],
            {"result": "ATLAS", "language": 1},
            False,
        ),
    ),
)
def test_output_fields_validate_required_optional_and_type(
    output_fields: list[dict[str, object]],
    output: Mapping[str, object],
    completed: bool,
) -> None:
    skill = _load_definition(output_fields=output_fields)

    result, executor = _execute(skill, "Atlas", output=output)

    assert result.completed is completed
    assert result.error_code == (
        None if completed else "SKILL_OUTPUT_CONTRACT_VIOLATION"
    )
    assert executor.calls == 1


@pytest.mark.parametrize(
    ("input_fields", "error_text"),
    (
        ([{"name": "value", "type": "unknown", "required": True}], "type"),
        (
            [
                {"name": "value", "type": "string", "required": True},
                {"name": "value", "type": "string", "required": False},
            ],
            "duplicate",
        ),
        ([{"name": "value", "type": "string", "required": "yes"}], "required"),
        ([{"name": "", "type": "string", "required": True}], "name"),
        (
            [
                {
                    "name": "value",
                    "type": "string",
                    "required": True,
                    "default": "Atlas",
                }
            ],
            "unknown",
        ),
    ),
)
def test_manifest_rejects_invalid_typed_fields(
    input_fields: list[dict[str, object]],
    error_text: str,
) -> None:
    loaded = SkillManifestLoader().load(
        json.dumps(_manifest(input_fields=input_fields))
    )

    assert loaded.status is SkillManifestStatus.INVALID
    assert error_text in " ".join(loaded.errors).lower()


@pytest.mark.parametrize(
    "overrides",
    (
        {"input_names": ["value"]},
        {"output_names": ["result"]},
    ),
)
def test_manifest_rejects_mixed_legacy_and_typed_contracts(
    overrides: Mapping[str, object],
) -> None:
    loaded = SkillManifestLoader().load(json.dumps(_manifest(**overrides)))

    assert loaded.status is SkillManifestStatus.INVALID
    assert "cannot be combined" in " ".join(loaded.errors).lower()


def test_legacy_name_contract_remains_required_and_untyped() -> None:
    legacy = _manifest(
        input_names=["value"],
        output_names=["result"],
    )
    legacy.pop("input_fields")
    legacy.pop("output_fields")
    loaded = SkillManifestLoader().load(json.dumps(legacy))
    assert loaded.status is SkillManifestStatus.VALID
    assert loaded.definition is not None
    skill = loaded.definition
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(registry, {"result": 42})

    valid = executor.execute(SkillExecutionRequest(skill, inputs={"value": 42}))
    missing = executor.execute(SkillExecutionRequest(skill, inputs={}))

    assert valid.status is SkillExecutionStatus.COMPLETED
    assert missing.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"


def test_typed_contracts_continue_to_allow_additional_fields() -> None:
    skill = _load_definition()
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(
        registry,
        {"result": "ATLAS", "additional_output": True},
    )

    result = executor.execute(
        SkillExecutionRequest(
            skill,
            inputs={"value": "Atlas", "additional_input": True},
        )
    )

    assert result.status is SkillExecutionStatus.COMPLETED
    assert result.output == {"additional_output": True, "result": "ATLAS"}


def test_authoritative_registered_typed_contract_overrides_caller_definition() -> None:
    registered = _load_definition()
    caller = SkillDefinition(
        skill_id=registered.skill_id,
        name="Caller definition",
        version="1.0",
        description="Untrusted caller definition without typed fields.",
        execution_target=registered.execution_target,
        execution_target_type=SkillExecutionTargetType.HANDLER,
        handler_id=registered.handler_id,
    )
    registry = SkillRegistry((registered,))
    executor = _RecordingExecutor(registry, {"result": "unused"})

    result = executor.execute(
        SkillExecutionRequest(caller, inputs={"value": 42})
    )

    assert result.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert executor.calls == 0


@pytest.mark.parametrize("target_type", tuple(SkillExecutionTargetType))
def test_typed_contract_validation_is_common_to_every_target(
    target_type: SkillExecutionTargetType,
) -> None:
    skill = _load_definition(execution_target_type=target_type.value)
    registry = SkillRegistry((skill,))
    executor = _RecordingExecutor(registry, {"result": 42})

    invalid_input = executor.execute(
        SkillExecutionRequest(skill, inputs={"value": 42})
    )
    invalid_output = executor.execute(
        SkillExecutionRequest(skill, inputs={"value": "Atlas"})
    )

    assert invalid_input.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert invalid_output.error_code == "SKILL_OUTPUT_CONTRACT_VIOLATION"
    assert executor.calls == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    (("Atlas", "ATLAS"), ("hola atlas", "HOLA ATLAS")),
)
def test_builtin_typed_manifest_executes_end_to_end(
    text: str,
    expected: str,
) -> None:
    built = build_core_agent_system(
        skill_handler_registry=build_builtin_skill_handler_registry(),
    )
    assert built.system is not None
    skill_system = built.system.skill_system

    registration = register_builtin_skills(skill_system)
    resolution = skill_system.skill_resolver.resolve(
        SkillResolutionRequest(required_skill_ids=("skill.text-uppercase",))
    )
    assert registration.registered_skill_ids == ("skill.text-uppercase",)
    assert resolution.status is SkillResolutionStatus.RESOLVED
    assert resolution.selected_skill is not None
    assert resolution.selected_skill.input_fields[0].type_name == "string"

    result = skill_system.skill_executor.execute(
        SkillExecutionRequest(resolution.selected_skill, inputs={"text": text})
    )

    assert result.status is SkillExecutionStatus.COMPLETED
    assert result.output == {"result": expected}
