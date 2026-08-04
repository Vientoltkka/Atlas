from __future__ import annotations

from collections.abc import Mapping
import json

from bootstrap.agent_system import build_core_agent_system
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
    AgentCooperationTaskStatus,
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
    SkillExecutionResult,
    SkillExecutionStatus,
    SkillHandlerRegistry,
)
from core.skill_registration import SkillRegistrationRequest, SkillRegistrationStatus
from core.skill_registry import SkillDefinition, SkillExecutionTargetType


class EchoAgentHandler:
    calls: list[Mapping[str, object]] = []

    @property
    def agent_id(self) -> str:
        return "agent.skill.runtime"

    def handle(self, context: AgentContext) -> Mapping[str, object]:
        self.calls.append(context.structured_input)
        return {"agent_result": context.structured_input["text"]}


def _agent(*, capabilities: tuple[str, ...] = ("text.transform",), denied: bool = False) -> AgentDefinition:
    metadata = {"denied_skill_ids": "skill.uppercase"} if denied else {"allowed_skill_ids": "skill.uppercase"}
    return AgentDefinition(
        agent_id="agent.skill.runtime",
        agent_type=AgentType.GENERAL,
        name="Skill runtime agent",
        description="Agent used by the productive skill runtime integration tests.",
        capabilities=AgentCapabilities(capabilities=capabilities),
        permissions=AgentPermissions(requires_confirmation=False),
        context_policy=AgentContextPolicy(allow_shared_context=True),
        metadata=metadata,
    )


def _manifest(*, enabled: bool = True, target: str = "handler.uppercase") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "skill_id": "skill.uppercase",
        "name": "Uppercase",
        "version": "1.0",
        "description": "Return dynamic text in uppercase.",
        "enabled": enabled,
        "required_capability_ids": ["text.transform"],
        "allowed_agent_types": ["general"],
        "input_names": ["text"],
        "output_names": ["result"],
        "execution_target": target,
        "execution_target_type": "handler",
        "handler_id": target,
    }


def _build_system(*, handler=None, agent: AgentDefinition | None = None):
    handlers = SkillHandlerRegistry()
    if handler is not None:
        handlers.register("handler.uppercase", handler)
    built = build_core_agent_system(skill_handler_registry=handlers)
    assert built.system is not None
    system = built.system
    selected_agent = agent or _agent()
    system.agent_registry.register(selected_agent)
    system.agent_handler_registry.register(EchoAgentHandler())
    return system


def _register_manifest(system, tmp_path, *, enabled: bool = True, target: str = "handler.uppercase") -> None:
    root = tmp_path / "skills"
    root.mkdir()
    (root / "uppercase.json").write_text(
        json.dumps(_manifest(enabled=enabled, target=target)),
        encoding="utf-8",
    )
    result = system.skill_system.skill_registration_service.register(
        SkillRegistrationRequest((str(root),))
    )
    assert result.status is SkillRegistrationStatus.COMPLETED


def _plan(skill_id: str | None = "skill.uppercase") -> AgentCooperationPlan:
    return AgentCooperationPlan(
        "plan.skill.runtime",
        (
            AgentCooperationTask(
                task_id="task.skill.runtime",
                objective_id="objective.skill.runtime",
                agent_id="agent.skill.runtime",
                structured_input={"text": "Atlas"},
                required_skill_ids=() if skill_id is None else (skill_id,),
            ),
        ),
    )


def _execute(system, plan: AgentCooperationPlan):
    return system.agent_cooperation_planner.execute(
        AgentCooperationPlanRequest(plan, policy=AgentCooperationPlanPolicy(enabled=True))
    )


def test_objective_planner_registered_skill_executes_with_dynamic_input(tmp_path) -> None:
    calls: list[Mapping[str, object]] = []
    system = _build_system(handler=lambda inputs: calls.append(inputs) or {"result": inputs["text"].upper()})
    _register_manifest(system, tmp_path)

    planning = system.agent_cooperation_automatic_planner.plan(
        AgentCooperationPlanningRequest(
            objective_id="objective.skill.runtime",
            objective_type=AgentCooperationObjectiveType.ANALYSIS,
            required_agent_ids=("agent.skill.runtime",),
            required_skill_ids=("skill.uppercase",),
            structured_input={"text": "Atlas"},
            policy=AgentCooperationPlanningPolicy(enabled=True),
        )
    )
    assert planning.status is AgentCooperationPlanningStatus.SUCCESS
    assert planning.plan is not None

    result = _execute(system, planning.plan)

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    task_id = planning.plan.tasks[0].task_id
    assert result.outputs[task_id]["result"] == "ATLAS"
    assert calls == [{"text": "Atlas"}]
    task_result = result.task_results[0]
    assert isinstance(task_result.execution_result, SkillExecutionResult)
    assert task_result.execution_result.status is SkillExecutionStatus.COMPLETED


def test_unregistered_skill_definition_cannot_execute() -> None:
    calls: list[Mapping[str, object]] = []
    system = _build_system(handler=lambda inputs: calls.append(inputs) or {"result": "unexpected"})
    external = SkillDefinition(
        skill_id="skill.uppercase",
        name="External",
        version="1.0",
        description="This definition is not registered.",
        execution_target="handler.uppercase",
        execution_target_type=SkillExecutionTargetType.HANDLER,
    )

    result = system.skill_system.skill_executor.execute(
        SkillExecutionRequest(external, inputs={"text": "Atlas"}, agent=_agent())
    )

    assert result.status is SkillExecutionStatus.TARGET_UNAVAILABLE
    assert result.error_code == "SKILL_NOT_REGISTERED"
    assert calls == []


def test_disabled_and_missing_skills_are_structured_and_not_executed(tmp_path) -> None:
    calls: list[Mapping[str, object]] = []
    system = _build_system(handler=lambda inputs: calls.append(inputs) or {"result": "unexpected"})
    _register_manifest(system, tmp_path, enabled=False)

    disabled = _execute(system, _plan())
    missing = _execute(system, _plan("skill.missing"))

    assert disabled.status is AgentCooperationPlanStatus.INVALID_PLAN
    assert disabled.error_code == "SKILL_DISABLED"
    assert missing.status is AgentCooperationPlanStatus.INVALID_PLAN
    assert missing.error_code == "SKILL_NOT_FOUND"
    assert calls == []


def test_missing_capability_and_denied_authorization_do_not_execute(tmp_path) -> None:
    capability_calls: list[Mapping[str, object]] = []
    capability_system = _build_system(
        handler=lambda inputs: capability_calls.append(inputs) or {"result": "unexpected"},
        agent=_agent(capabilities=()),
    )
    _register_manifest(capability_system, tmp_path)

    capability_result = _execute(capability_system, _plan())

    denied_calls: list[Mapping[str, object]] = []
    denied_system = _build_system(
        handler=lambda inputs: denied_calls.append(inputs) or {"result": "unexpected"},
        agent=_agent(denied=True),
    )
    denied_system.skill_system.skill_registry.register(
        capability_system.skill_system.skill_registry.get("skill.uppercase")
    )
    denied_result = _execute(denied_system, _plan())

    assert capability_result.status is AgentCooperationPlanStatus.FAILED
    assert capability_result.task_results[0].status is AgentCooperationTaskStatus.SKILL_AUTHORIZATION_FAILED
    assert denied_result.status is AgentCooperationPlanStatus.FAILED
    assert denied_result.task_results[0].status is AgentCooperationTaskStatus.SKILL_AUTHORIZATION_FAILED
    assert capability_calls == []
    assert denied_calls == []


def test_target_failure_is_propagated_through_skill_result(tmp_path) -> None:
    def fail(_inputs):
        raise RuntimeError("controlled target failure")

    system = _build_system(handler=fail)
    _register_manifest(system, tmp_path)

    result = _execute(system, _plan())

    assert result.status is AgentCooperationPlanStatus.FAILED
    task_result = result.task_results[0]
    assert task_result.status is AgentCooperationTaskStatus.SKILL_EXECUTION_FAILED
    assert isinstance(task_result.execution_result, SkillExecutionResult)
    assert task_result.execution_result.status is SkillExecutionStatus.EXECUTION_FAILED
    assert task_result.error_code == "RuntimeError"


def test_task_without_skill_keeps_agent_execution_behavior() -> None:
    EchoAgentHandler.calls = []
    system = _build_system()

    result = _execute(system, _plan(None))

    assert result.status is AgentCooperationPlanStatus.SUCCESS
    assert result.outputs["task.skill.runtime"]["agent_result"] == "Atlas"
    assert EchoAgentHandler.calls == [{"text": "Atlas"}]
