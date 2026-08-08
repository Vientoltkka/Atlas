from __future__ import annotations

from collections.abc import Mapping

import pytest

from bootstrap.agent_system import build_core_agent_system
from bootstrap.bootstrap import Bootstrap
from bootstrap.skill_system import (
    BUILTIN_SKILLS_ROOT,
    build_builtin_skill_handler_registry,
    register_builtin_skills,
)
from core.agent_context import AgentContext
from core.agent_cooperation_automatic_planner import (
    AgentCooperationObjectiveType,
    AgentCooperationPlanningPolicy,
    AgentCooperationPlanningRequest,
    AgentCooperationPlanningStatus,
)
from core.agent_cooperation_plan import (
    AgentCooperationPlan,
    AgentCooperationPlanPolicy,
    AgentCooperationPlanRequest,
    AgentCooperationPlanStatus,
    AgentCooperationTask,
)
from core.agent_registry import (
    AgentCapabilities,
    AgentDefinition,
    AgentPermissions,
    AgentType,
)
from core.skill_discovery import SkillDiscoveryRequest, SkillDiscoveryStatus
from core.skill_executor import SkillExecutionRequest, SkillExecutionStatus
from core.skill_manifest import SkillManifestStatus
from core.skill_registration import SkillRegistrationStatus
from core.skill_registry import SkillDefinition, SkillExecutionTargetType
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus


SKILL_ID = "skill.text-uppercase"
AGENT_ID = "agent.builtin-skill-test"


class _EchoAgentHandler:
    calls: list[Mapping[str, object]] = []

    @property
    def agent_id(self) -> str:
        return AGENT_ID

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        self.calls.append(context.structured_input)
        return {"agent_result": context.structured_input["text"]}


def _agent() -> AgentDefinition:
    return AgentDefinition(
        agent_id=AGENT_ID,
        agent_type=AgentType.GENERAL,
        name="Builtin skill test agent",
        description="Agent used to verify builtin skill routing.",
        capabilities=AgentCapabilities(),
        permissions=AgentPermissions(requires_confirmation=False),
        metadata={"allowed_skill_ids": SKILL_ID},
    )


def _runtime():
    result = build_core_agent_system(
        skill_handler_registry=build_builtin_skill_handler_registry(),
    )
    assert result.system is not None
    system = result.system
    registration = register_builtin_skills(system.skill_system)
    system.agent_registry.register(_agent())
    system.agent_handler_registry.register(_EchoAgentHandler())
    return system, registration


def _plan(text: str, *, skill_id: str | None = SKILL_ID) -> AgentCooperationPlan:
    return AgentCooperationPlan(
        "plan.builtin-skill",
        (
            AgentCooperationTask(
                task_id="task.builtin-skill",
                objective_id="objective.builtin-skill",
                agent_id=AGENT_ID,
                structured_input={"text": text},
                required_skill_ids=() if skill_id is None else (skill_id,),
            ),
        ),
    )


def _execute(system, plan: AgentCooperationPlan):
    return system.agent_cooperation_planner.execute(
        AgentCooperationPlanRequest(
            plan,
            policy=AgentCooperationPlanPolicy(enabled=True),
        )
    )


def test_productive_discovery_finds_and_loads_the_builtin_manifest() -> None:
    system, _ = _runtime()

    discovery = system.skill_system.skill_discovery.discover(
        SkillDiscoveryRequest((BUILTIN_SKILLS_ROOT,), recursive=True)
    )

    assert discovery.status is SkillDiscoveryStatus.COMPLETED
    assert [manifest.path.name for manifest in discovery.manifests] == ["skill.json"]
    loaded = system.skill_system.skill_manifest_loader.load(discovery.manifests[0].content)
    assert loaded.status is SkillManifestStatus.VALID
    assert loaded.definition is not None
    assert loaded.definition.skill_id == SKILL_ID


def test_real_bootstrap_registers_builtin_skill_in_the_central_registry() -> None:
    orchestrator = Bootstrap.build()
    assert orchestrator._atlas_router is not None
    agent_system = orchestrator._atlas_router._agent_system
    assert agent_system is not None

    skill = agent_system.skill_system.skill_registry.get(SKILL_ID)
    resolution = agent_system.skill_system.skill_resolver.resolve(
        SkillResolutionRequest(required_skill_ids=(SKILL_ID,))
    )

    assert skill.handler_id == "handler.text-uppercase"
    assert resolution.status is SkillResolutionStatus.RESOLVED
    assert resolution.selected_skill is skill
    assert (
        agent_system.skill_system.skill_executor.skill_registry
        is agent_system.skill_system.skill_registry
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    (("Atlas", "ATLAS"), ("hola atlas", "HOLA ATLAS")),
)
def test_automatic_planner_executes_builtin_skill_with_dynamic_input(
    text: str,
    expected: str,
) -> None:
    system, _ = _runtime()
    planning = system.agent_cooperation_automatic_planner.plan(
        AgentCooperationPlanningRequest(
            objective_id="objective.builtin-skill",
            objective_type=AgentCooperationObjectiveType.ANALYSIS,
            required_agent_ids=(AGENT_ID,),
            required_skill_ids=(SKILL_ID,),
            structured_input={"text": text},
            policy=AgentCooperationPlanningPolicy(enabled=True),
        )
    )

    assert planning.status is AgentCooperationPlanningStatus.SUCCESS
    assert planning.plan is not None
    execution = _execute(system, planning.plan)

    assert execution.status is AgentCooperationPlanStatus.SUCCESS
    assert execution.outputs[planning.plan.tasks[0].task_id] == {"result": expected}
    skill_result = execution.task_results[0].execution_result
    assert skill_result is not None
    assert skill_result.status is SkillExecutionStatus.COMPLETED


def test_real_skills_block_runs_end_to_end_with_productive_handler() -> None:
    system, registration = _runtime()
    registered = system.skill_system.skill_registry.get(SKILL_ID)
    resolution = system.skill_system.skill_resolver.resolve(
        SkillResolutionRequest(required_skill_ids=(SKILL_ID,))
    )
    planning = system.agent_cooperation_automatic_planner.plan(
        AgentCooperationPlanningRequest(
            objective_id="objective.skills-e2e",
            objective_type=AgentCooperationObjectiveType.ANALYSIS,
            required_agent_ids=(AGENT_ID,),
            required_skill_ids=(SKILL_ID,),
            structured_input={"text": "Atlas end to end"},
            policy=AgentCooperationPlanningPolicy(enabled=True),
        )
    )

    assert registration.status is SkillRegistrationStatus.COMPLETED
    assert registration.registered_skill_ids == (SKILL_ID,)
    assert registered.execution_target_type is SkillExecutionTargetType.HANDLER
    assert registered.handler_id == "handler.text-uppercase"
    assert resolution.status is SkillResolutionStatus.RESOLVED
    assert resolution.selected_skill is registered
    assert planning.status is AgentCooperationPlanningStatus.SUCCESS
    assert planning.plan is not None
    assert planning.available_skill_ids == (SKILL_ID,)
    assert planning.plan.tasks[0].required_skill_ids == (SKILL_ID,)

    execution = _execute(system, planning.plan)

    assert execution.status is AgentCooperationPlanStatus.SUCCESS
    assert execution.outputs[planning.plan.tasks[0].task_id] == {
        "result": "ATLAS END TO END"
    }
    task_result = execution.task_results[0]
    skill_result = task_result.execution_result
    assert skill_result is not None
    assert skill_result.status is SkillExecutionStatus.COMPLETED
    assert skill_result.output == {"result": "ATLAS END TO END"}
    assert [event.name for event in skill_result.events] == [
        "skill_execution_started",
        "skill_execution_succeeded",
    ]


def test_real_text_entry_executes_builtin_skill_with_dynamic_input(
    monkeypatch,
    capsys,
) -> None:
    orchestrator = Bootstrap.build()
    inputs = iter(
        (
            'Usa la Skill Text Uppercase con el texto "Atlas dinamico 20.2"',
            "salir",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    orchestrator.start()

    visible = capsys.readouterr().out
    assert "Atlas:" in visible
    assert "ATLAS DINAMICO 20.2" in visible
    assert "Hasta pronto." in visible


def test_missing_disabled_and_unregistered_skills_do_not_execute() -> None:
    system, _ = _runtime()

    missing = _execute(system, _plan("Atlas", skill_id="skill.missing"))
    registered = system.skill_system.skill_registry.get(SKILL_ID)
    invalid = system.skill_system.skill_executor.execute(
        SkillExecutionRequest(registered, inputs={"text": 7}, agent=_agent())
    )
    system.skill_system.skill_registry.register(
        SkillDefinition(
            skill_id=registered.skill_id,
            name=registered.name,
            version=registered.version,
            description=registered.description,
            enabled=False,
            input_names=registered.input_names,
            output_names=registered.output_names,
            execution_target=registered.execution_target,
            execution_target_type=registered.execution_target_type,
            handler_id=registered.handler_id,
        ),
        replace=True,
    )
    disabled = system.skill_system.skill_executor.execute(
        SkillExecutionRequest(registered, inputs={"text": "Atlas"}, agent=_agent())
    )
    external = SkillDefinition(
        skill_id="skill.external-uppercase",
        name="External uppercase",
        version="1.0",
        description="Unregistered external definition.",
        execution_target="handler.text-uppercase",
        execution_target_type=SkillExecutionTargetType.HANDLER,
        handler_id="handler.text-uppercase",
    )
    unregistered = system.skill_system.skill_executor.execute(
        SkillExecutionRequest(external, inputs={"text": "Atlas"}, agent=_agent())
    )

    assert missing.status is AgentCooperationPlanStatus.INVALID_PLAN
    assert missing.error_code == "SKILL_NOT_FOUND"
    assert invalid.status is SkillExecutionStatus.EXECUTION_FAILED
    assert invalid.error_code == "SKILL_INPUT_CONTRACT_VIOLATION"
    assert disabled.status is SkillExecutionStatus.SKILL_DISABLED
    assert disabled.output is None
    assert unregistered.status is SkillExecutionStatus.TARGET_UNAVAILABLE
    assert unregistered.error_code == "SKILL_NOT_REGISTERED"


def test_task_without_skill_still_uses_agent_executor() -> None:
    _EchoAgentHandler.calls = []
    system, _ = _runtime()

    result = _execute(system, _plan("sin skill", skill_id=None))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.outputs["task.builtin-skill"] == {"agent_result": "sin skill"}
    assert _EchoAgentHandler.calls == [{"text": "sin skill"}]


def test_repeated_builtin_registration_keeps_one_authoritative_definition() -> None:
    system, first = _runtime()

    second = register_builtin_skills(system.skill_system)

    assert first.status is SkillRegistrationStatus.COMPLETED
    assert first.registered_skill_ids == (SKILL_ID,)
    assert second.status is SkillRegistrationStatus.COMPLETED
    assert second.registered_skill_ids == ()
    assert second.skipped_skill_ids == (SKILL_ID,)
    assert [
        skill.skill_id
        for skill in system.skill_system.skill_registry.list_skills()
    ] == [SKILL_ID]
