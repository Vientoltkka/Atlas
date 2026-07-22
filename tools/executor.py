"""Tool Executor."""

from __future__ import annotations

from collections.abc import Mapping

from core.execution_arguments import contains_unresolved_execution_reference
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext
from tools.tool_schema import ToolSchemaValidationException


class ToolExecutor:
    """Executes registered Atlas tools."""

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:

        self._registry = registry

    def execute(
        self,
        tool_name: str,
        context: ToolContext | None = None,
        *,
        arguments: Mapping[str, object] | None = None,
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
            active_context = ToolContext(parameters=normalized_arguments)
        else:
            active_context = ToolContext(
                parameters=normalized_arguments,
                step_id=context.step_id,
                plan_signature=context.plan_signature,
                previous_results=context.previous_results,
                metadata=context.metadata,
            )

        return tool.execute(active_context)
