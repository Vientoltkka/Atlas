"""Factory for Atlas declarative skill system."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from core.agent_executor import AgentExecutor
from core.capability_execution_service import CapabilityExecutionService
from core.skill_discovery import SkillDiscovery
from core.skill_executor import SkillExecutor, SkillHandlerRegistry
from core.skill_manifest import SkillManifestLoader
from core.skill_registration import (
    SkillDuplicatePolicy,
    SkillRegistrationPolicy,
    SkillRegistrationRequest,
    SkillRegistrationResult,
    SkillRegistrationService,
    SkillRegistrationStatus,
)
from core.skill_execution_context import SkillExecutionContext
from core.skill_registry import SkillRegistry
from core.skill_resolver import SkillResolver
from core.skill_system import SkillSystem, build_skill_system
from tools.executor import ToolExecutor
from use_cases.desktop_layout_skills import WindowLayoutSkills


BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills" / "builtin"
DESKTOP_SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills" / "desktop"
TEXT_UPPERCASE_HANDLER_ID = "handler.text-uppercase"
MODO_TRABAJO_HANDLER_ID = "handler.modo-trabajo"


def build_core_skill_system(
    *,
    skill_registry: SkillRegistry | None = None,
    skill_manifest_loader: SkillManifestLoader | None = None,
    skill_discovery: SkillDiscovery | None = None,
    skill_registration_service: SkillRegistrationService | None = None,
    skill_resolver: SkillResolver | None = None,
    skill_executor: SkillExecutor | None = None,
    tool_executor: ToolExecutor | None = None,
    capability_execution_service: CapabilityExecutionService | None = None,
    agent_executor: AgentExecutor | None = None,
    skill_handler_registry: SkillHandlerRegistry | None = None,
) -> SkillSystem:
    return build_skill_system(
        skill_registry=skill_registry,
        skill_manifest_loader=skill_manifest_loader,
        skill_discovery=skill_discovery,
        skill_registration_service=skill_registration_service,
        skill_resolver=skill_resolver,
        skill_executor=skill_executor,
        tool_executor=tool_executor,
        capability_execution_service=capability_execution_service,
        agent_executor=agent_executor,
        skill_handler_registry=skill_handler_registry,
    )


def build_builtin_skill_handler_registry(
    tool_executor: ToolExecutor | None = None,
) -> SkillHandlerRegistry:
    """Build the explicit registry for handlers shipped with Atlas."""

    registry = SkillHandlerRegistry()
    registry.register(TEXT_UPPERCASE_HANDLER_ID, _text_uppercase_handler)
    layout_skills = WindowLayoutSkills(tool_executor)
    registry.register(MODO_TRABAJO_HANDLER_ID, layout_skills.modo_trabajo)
    return registry


def register_builtin_skills(skill_system: SkillSystem) -> SkillRegistrationResult:
    """Discover and register the declarative builtin manifests."""

    return _register_manifests(skill_system, BUILTIN_SKILLS_ROOT)


def register_desktop_skills(skill_system: SkillSystem) -> SkillRegistrationResult:
    """Discover and register the declarative desktop layout manifests."""

    return _register_manifests(skill_system, DESKTOP_SKILLS_ROOT)


def _register_manifests(
    skill_system: SkillSystem,
    root: Path,
) -> SkillRegistrationResult:
    if not isinstance(skill_system, SkillSystem):
        raise TypeError("skill_system must be SkillSystem.")
    result = skill_system.skill_registration_service.register(
        SkillRegistrationRequest(
            (str(root),),
            recursive=True,
            policy=SkillRegistrationPolicy(
                duplicate_policy=SkillDuplicatePolicy.KEEP_EXISTING,
            ),
        )
    )
    if result.status is not SkillRegistrationStatus.COMPLETED:
        raise RuntimeError("builtin skill registration failed")
    return result


def _text_uppercase_handler(
    inputs: Mapping[str, object],
    *,
    execution_context: SkillExecutionContext,
) -> Mapping[str, object]:
    if execution_context.is_cancelled or execution_context.remaining_seconds is None:
        raise RuntimeError("text uppercase execution context is unavailable")
    text = inputs.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return {"result": text.upper()}
