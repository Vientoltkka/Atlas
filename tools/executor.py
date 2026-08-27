"""Tool Executor."""

from __future__ import annotations

from collections.abc import Mapping

from core.execution_arguments import contains_unresolved_execution_reference
from core.skill_execution_context import SkillExecutionContext
from tools.effect_permissions import (
    ToolEffectAuthorization,
    ToolEffectPermissionPolicy,
)
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext
from tools.tool_schema import ToolSchemaValidationException


class ToolExecutor:
    """Executes registered Atlas tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        permission_policy: ToolEffectPermissionPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._permission_policy = permission_policy or ToolEffectPermissionPolicy()

    def requires_explicit_authorization(self, tool_name: str) -> bool:
        return self._permission_policy.requires_authorization(
            getattr(self._registry.get(tool_name), "required_permissions", ())
        )

    def authorize(self, tool_name: str) -> ToolEffectAuthorization:
        tool = self._registry.get(tool_name)
        return self._permission_policy.authorize(tool_name, getattr(tool, "required_permissions", ()))

    def execute(
        self,
        tool_name: str,
        context: ToolContext | None = None,
        *,
        arguments: Mapping[str, object] | None = None,
        execution_context: SkillExecutionContext | None = None,
        authorization: ToolEffectAuthorization | None = None,
    ):
        """Execute a registered tool."""

        tool = self._registry.get(tool_name)
        source_arguments = (
            context.parameters
            if context is not None
            else dict(arguments or {})
        )
        if contains_unresolved_execution_reference(source_arguments):
            raise ValueError(
                "ToolExecutor received unresolved StepOutputReference or "
                "ExecutionVariableReference arguments."
            )

        schema = self._registry.arguments_schema(tool_name)
        if schema is None:
            normalized_arguments = dict(source_arguments)
        else:
            validation = schema.validate(tool_name, source_arguments)
            if not validation.is_valid:
                raise ToolSchemaValidationException(validation)
            normalized_arguments = dict(validation.normalized_arguments)

        if context is None:
            active_context = ToolContext(parameters=normalized_arguments, execution_context=execution_context)
        else:
            active_context = ToolContext(
                parameters=normalized_arguments,
                step_id=context.step_id,
                plan_signature=context.plan_signature,
                previous_results=context.previous_results,
                metadata=context.metadata,
                execution_context=execution_context or context.execution_context,
            )

        self._permission_policy.require(
            tool_name,
            getattr(tool, "required_permissions", ()),
            authorization,
        )
        return tool.execute(active_context)
