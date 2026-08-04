from __future__ import annotations

import json

import pytest

from bootstrap.agent_system import build_core_agent_system
from core.agent_registry import AgentCapabilities, AgentDefinition, AgentPermissions, AgentType
from core.skill_discovery import SkillDiscovery, SkillDiscoveryRequest, SkillDiscoveryStatus
from core.skill_executor import SkillExecutionRequest, SkillExecutionStatus, SkillExecutor, SkillHandlerRegistry
from core.skill_manifest import SkillManifestLoader, SkillManifestStatus
from core.skill_registration import SkillDuplicatePolicy, SkillRegistrationPolicy, SkillRegistrationRequest, SkillRegistrationService, SkillRegistrationStatus
from core.skill_registry import InvalidSkillDefinitionError, SkillDefinition, SkillExecutionTargetType, SkillRegistry, SkillLimits
from core.skill_resolver import SkillResolutionRequest, SkillResolutionStatus, SkillResolver
from core.skill_system import build_skill_system
from tools.executor import ToolExecutor
from tools.filesystem.list_directory_tool import ListDirectoryTool
from tools.registry import ToolRegistry


def _manifest(**overrides):
    data = {
        "schema_version": "1.0",
        "skill_id": "skill.list_directory",
        "name": "List directory",
        "version": "1.0",
        "description": "List directory entries.",
        "enabled": True,
        "required_capability_ids": ("filesystem.list",),
        "required_permission_ids": ("can_read_project",),
        "allowed_agent_types": ("general",),
        "input_names": ("path",),
        "output_names": ("result",),
        "execution_target": "list_directory",
        "execution_target_type": "tool",
        "limits": {"timeout_seconds": 10, "max_inputs": 4, "max_outputs": 4, "max_result_items": 16},
        "metadata": {"source": "test"},
        "tags": ("filesystem",),
    }
    data.update(overrides)
    return data


def _skill(skill_id: str = "skill.demo", **overrides) -> SkillDefinition:
    data = {
        "skill_id": skill_id,
        "name": skill_id,
        "version": "1.0",
        "description": "Demo skill.",
        "required_capability_ids": ("demo.capability",),
        "required_permission_ids": ("can_read_project",),
        "allowed_agent_types": (AgentType.GENERAL,),
        "input_names": ("value",),
        "output_names": ("result",),
        "execution_target": "handler.demo",
        "execution_target_type": SkillExecutionTargetType.HANDLER,
        "tags": ("demo",),
    }
    data.update(overrides)
    return SkillDefinition(**data)


def _agent(metadata=None) -> AgentDefinition:
    return AgentDefinition(
        agent_id="agent.demo",
        agent_type=AgentType.GENERAL,
        name="agent.demo",
        description="Demo agent.",
        capabilities=AgentCapabilities(capabilities=("demo.capability",)),
        permissions=AgentPermissions(requires_confirmation=False),
        metadata=metadata or {},
    )


def test_skill_definition_valid_and_invalid_values() -> None:
    skill = _skill()

    assert skill.skill_id == "skill.demo"
    assert skill.execution_target_type is SkillExecutionTargetType.HANDLER
    with pytest.raises(InvalidSkillDefinitionError):
        _skill("__class__")
    with pytest.raises(InvalidSkillDefinitionError):
        _skill(version="v1")
    with pytest.raises(InvalidSkillDefinitionError):
        _skill(metadata={"api_key": "hidden"})
    with pytest.raises(InvalidSkillDefinitionError):
        _skill(limits=SkillLimits(timeout_seconds=0))


def test_manifest_loader_valid_invalid_json_unknown_sensitive_and_limits() -> None:
    loader = SkillManifestLoader()
    valid = loader.load(json.dumps(_manifest()))

    assert valid.status is SkillManifestStatus.VALID
    assert valid.definition is not None
    assert valid.definition.skill_id == "skill.list_directory"
    assert loader.load("{bad json").status is SkillManifestStatus.INVALID
    assert loader.load(json.dumps(_manifest(extra=True))).status is SkillManifestStatus.INVALID
    assert loader.load(json.dumps(_manifest(metadata={"password": "hidden"}))).status is SkillManifestStatus.INVALID
    assert loader.load(json.dumps(_manifest(limits={"timeout_seconds": 0}))).status is SkillManifestStatus.INVALID
    assert loader.load(json.dumps(_manifest(metadata={"x": float("nan")}))).status is SkillManifestStatus.INVALID


def test_registry_deterministic_queries_and_duplicates() -> None:
    registry = SkillRegistry()
    coding = _skill("skill.coding", tags=("review",), allowed_agent_types=(AgentType.CODING,))
    general = _skill("skill.general", tags=("review", "demo"))

    registry.register(coding)
    registry.register(general)

    assert registry.contains("skill.coding")
    assert [skill.skill_id for skill in registry.list_skills()] == ["skill.coding", "skill.general"]
    assert registry.find_by_capability("demo.capability") == (coding, general)
    assert registry.find_by_tag("demo") == (general,)
    assert registry.find_by_agent_type(AgentType.CODING) == (coding,)
    with pytest.raises(Exception):
        registry.register(coding)


def test_discovery_safe_order_hidden_and_symlink(tmp_path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    (root / "b.json").write_text(json.dumps(_manifest(skill_id="skill.b")), encoding="utf-8")
    (root / "a.json").write_text(json.dumps(_manifest(skill_id="skill.a")), encoding="utf-8")
    (root / ".hidden.json").write_text(json.dumps(_manifest(skill_id="skill.hidden")), encoding="utf-8")
    try:
        (root / "link.json").symlink_to(root / "a.json")
    except OSError:
        pass

    result = SkillDiscovery().discover(SkillDiscoveryRequest((root,)))

    assert result.status is SkillDiscoveryStatus.COMPLETED
    assert [manifest.path.name for manifest in result.manifests] == ["a.json", "b.json"]


def test_registration_dry_run_duplicates_and_atomic_rollback(tmp_path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    (root / "a.json").write_text(json.dumps(_manifest(skill_id="skill.a")), encoding="utf-8")
    registry = SkillRegistry()
    service = SkillRegistrationService(SkillDiscovery(), SkillManifestLoader(), registry)

    dry = service.register(SkillRegistrationRequest((str(root),), policy=SkillRegistrationPolicy(dry_run=True)))
    assert dry.status is SkillRegistrationStatus.DRY_RUN_COMPLETED
    assert not registry.contains("skill.a")

    done = service.register(SkillRegistrationRequest((str(root),)))
    duplicate = service.register(SkillRegistrationRequest((str(root),)))

    assert done.status is SkillRegistrationStatus.COMPLETED
    assert duplicate.status is SkillRegistrationStatus.REGISTRATION_FAILED
    assert registry.contains("skill.a")

    (root / "bad.json").write_text(json.dumps(_manifest(skill_id="skill.bad", metadata={"token": "x"})), encoding="utf-8")
    failed = service.register(SkillRegistrationRequest((str(root),), policy=SkillRegistrationPolicy(duplicate_policy=SkillDuplicatePolicy.REPLACE)))
    assert failed.status is SkillRegistrationStatus.REGISTRATION_FAILED
    assert registry.contains("skill.a")
    assert not registry.contains("skill.bad")


def test_resolver_by_id_capability_tag_agent_type_disabled_no_match_and_ambiguity() -> None:
    registry = SkillRegistry((
        _skill("skill.a", tags=("review",), allowed_agent_types=(AgentType.GENERAL,)),
        _skill("skill.b", tags=("review",), allowed_agent_types=(AgentType.GENERAL,)),
        _skill("skill.disabled", enabled=False),
    ))
    resolver = SkillResolver(registry)

    by_id = resolver.resolve(SkillResolutionRequest(required_skill_ids=("skill.a",)))
    by_cap = resolver.resolve(SkillResolutionRequest(required_capability_ids=("demo.capability",), preferred_tags=("review",), require_unique_top_score=False))
    by_tag = resolver.resolve(SkillResolutionRequest(required_tags=("review",), preferred_capability_ids=("demo.capability",), require_unique_top_score=False))
    by_type = resolver.resolve(SkillResolutionRequest(required_agent_types=(AgentType.GENERAL,), preferred_tags=("review",), require_unique_top_score=False))
    disabled = resolver.resolve(SkillResolutionRequest(required_skill_ids=("skill.disabled",)))
    no_match = resolver.resolve(SkillResolutionRequest(required_capability_ids=("missing.capability",)))
    ambiguous = resolver.resolve(SkillResolutionRequest(required_tags=("review",)))

    assert by_id.status is SkillResolutionStatus.RESOLVED
    assert by_cap.status is SkillResolutionStatus.RESOLVED
    assert by_tag.status is SkillResolutionStatus.RESOLVED
    assert by_type.status is SkillResolutionStatus.RESOLVED
    assert disabled.status is SkillResolutionStatus.NO_MATCHING_SKILL
    assert no_match.status is SkillResolutionStatus.NO_MATCHING_SKILL
    assert ambiguous.status is SkillResolutionStatus.AMBIGUOUS


def test_executor_handler_authorization_target_errors_and_sanitized_failure() -> None:
    handlers = SkillHandlerRegistry()
    handlers.register("handler.demo", lambda inputs: {"result": inputs["value"]})
    skill = _skill(handler_id="handler.demo")
    failing_skill = _skill("skill.fail", handler_id="handler.fail")
    registry = SkillRegistry((skill, failing_skill))
    executor = SkillExecutor(skill_registry=registry, handler_registry=handlers)

    ok = executor.execute(SkillExecutionRequest(skill, inputs={"value": "ok"}, agent=_agent(metadata={"allowed_skill_ids": "skill.demo"})))
    denied = executor.execute(SkillExecutionRequest(skill, inputs={"value": "ok"}, agent=_agent(metadata={"denied_skill_ids": "skill.demo"})))
    missing = SkillExecutor(skill_registry=registry).execute(SkillExecutionRequest(skill, inputs={"value": "ok"}))
    handlers.register("handler.fail", lambda _inputs: (_ for _ in ()).throw(RuntimeError("api_key leaked")))
    failed = executor.execute(SkillExecutionRequest(failing_skill, inputs={}))

    assert ok.status is SkillExecutionStatus.COMPLETED
    assert dict(ok.output) == {"result": "ok"}
    assert denied.status is SkillExecutionStatus.SKILL_NOT_AUTHORIZED
    assert missing.status is SkillExecutionStatus.EXECUTION_FAILED
    assert failed.status is SkillExecutionStatus.EXECUTION_FAILED
    assert "api_key" not in repr(failed)


def test_e2e_manifest_to_registry_resolver_executor_real_tool(tmp_path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / "a.txt").write_text("a", encoding="utf-8")
    (target / "b.txt").write_text("b", encoding="utf-8")
    (root / "list.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    tool_registry = ToolRegistry()
    tool_registry.register(ListDirectoryTool())
    skill_system = build_skill_system(skill_executor=SkillExecutor(tool_executor=ToolExecutor(tool_registry)))

    registered = skill_system.skill_registration_service.register(SkillRegistrationRequest((str(root),)))
    resolved = skill_system.skill_resolver.resolve(SkillResolutionRequest(required_skill_ids=("skill.list_directory",)))
    executed = skill_system.skill_executor.execute(
        SkillExecutionRequest(resolved.selected_skill, inputs={"path": str(target)}, agent=_agent(metadata={"allowed_skill_ids": "skill.list_directory"}))
    )

    assert registered.status is SkillRegistrationStatus.COMPLETED
    assert resolved.status is SkillResolutionStatus.RESOLVED
    assert executed.status is SkillExecutionStatus.COMPLETED
    assert executed.output["result"] == ("a.txt", "b.txt")
    assert executed.metrics["skill_executions_succeeded"] == 1


def test_agent_system_contains_skill_system_and_existing_agent_features_remain() -> None:
    result = build_core_agent_system()

    assert result.system is not None
    assert result.system.skill_system.skill_registry.list_skills() == ()
    assert result.system.multi_agent_coordinator is not None
