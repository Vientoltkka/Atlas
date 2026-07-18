"""Conversational adapter for structured execution coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.execution_coordinator import (
    ExecutionCoordinationResult,
    ExecutionCoordinationStatus,
    ExecutionCoordinator,
)
from tools.execution_decision import ExecutionMode
from tools.single_tool_runner import ToolRunResult
from tools.tool_chain_runner import ToolChainResult, ToolChainStepResult


@dataclass(frozen=True, slots=True)
class ExecutionConversationOutcome:
    """Result of routing one conversational turn through execution handling."""

    direct_response_required: bool
    text: str
    result: ExecutionCoordinationResult


class ExecutionConversationController:
    """Keep session confirmation state around ExecutionCoordinator."""

    def __init__(
        self,
        coordinator: ExecutionCoordinator,
        presenter: "ExecutionResultPresenter | None" = None,
    ) -> None:
        self._coordinator = coordinator
        self._presenter = presenter or ExecutionResultPresenter()
        self._pending_confirmation_id: str | None = None
        self._last_result: ExecutionCoordinationResult | None = None

    @property
    def pending_confirmation_id(self) -> str | None:
        """Return the active confirmation id for debugging."""
        return self._pending_confirmation_id

    @property
    def last_result(self) -> ExecutionCoordinationResult | None:
        """Return the latest structured result for debugging."""
        return self._last_result

    def handle(
        self,
        prompt: str,
    ) -> ExecutionConversationOutcome:
        """Handle one text turn before the normal conversational fallback."""
        if self._pending_confirmation_id is not None:
            result = self._coordinator.confirm(
                self._pending_confirmation_id,
                prompt,
            )
            self._remember(result)
            return ExecutionConversationOutcome(
                direct_response_required=False,
                text=self._presenter.present(result),
                result=result,
            )

        result = self._coordinator.execute(prompt)
        self._remember(result)

        if result.status is ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED:
            return ExecutionConversationOutcome(
                direct_response_required=True,
                text="",
                result=result,
            )

        return ExecutionConversationOutcome(
            direct_response_required=False,
            text=self._presenter.present(result),
            result=result,
        )

    def _remember(
        self,
        result: ExecutionCoordinationResult,
    ) -> None:
        self._last_result = result

        if result.status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED:
            self._pending_confirmation_id = result.confirmation_id
            return

        self._pending_confirmation_id = None


class ExecutionResultPresenter:
    """Convert structured execution results into user-facing text."""

    def present(
        self,
        result: ExecutionCoordinationResult,
    ) -> str:
        """Return a concise text response without dumping internal dataclasses."""
        if result.status is ExecutionCoordinationStatus.EXECUTED:
            return self._present_executed(result)

        if result.status is ExecutionCoordinationStatus.INFORMATION_REQUIRED:
            return _missing_information_message(result.missing_information)

        if result.status is ExecutionCoordinationStatus.AMBIGUOUS_REQUEST:
            return _ambiguous_information_message(result.ambiguous_information)

        if result.status is ExecutionCoordinationStatus.UNSUPPORTED:
            return "Atlas todavia no dispone de la herramienta necesaria para esa accion."

        if result.status is ExecutionCoordinationStatus.VALIDATION_FAILED:
            return _validation_failed_message(result.validation_errors)

        if result.status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED:
            return self._confirmation_message(result)

        if result.status is ExecutionCoordinationStatus.CANCELLED:
            return "Operacion cancelada."

        if result.status is ExecutionCoordinationStatus.FAILED:
            return "No he podido completar la operacion."

        return result.message

    def _present_executed(
        self,
        result: ExecutionCoordinationResult,
    ) -> str:
        execution_result = result.execution_result

        if isinstance(execution_result, ToolChainResult):
            return self._present_chain(execution_result)

        if isinstance(execution_result, ToolRunResult):
            return self._present_tool_result(execution_result)

        return "Operacion completada."

    def _present_chain(
        self,
        chain: ToolChainResult,
    ) -> str:
        lines = ["Cadena completada:"]

        for step in chain.steps:
            lines.append(f"- {step.step_id}: {_tool_label(step.tool_name)}")

        if chain.steps:
            final = self._present_tool_result(chain.steps[-1].result)
            lines.append("Resultado final:")
            lines.append(final)

        return "\n".join(lines)

    def _present_tool_result(
        self,
        result: ToolRunResult,
    ) -> str:
        if result.tool_name == "read_file":
            return str(result.result)

        if result.tool_name == "write_file":
            path = _argument(result, "path")
            if path:
                return f"Archivo escrito: {path}"
            return "Archivo escrito correctamente."

        if result.tool_name and result.tool_name.startswith("desktop."):
            return f"Accion completada: {result.result}"

        if isinstance(result.result, list | tuple):
            return _format_items(result.result)

        if result.result is not None:
            return str(result.result)

        return "Operacion completada."

    def _confirmation_message(
        self,
        result: ExecutionCoordinationResult,
    ) -> str:
        execution_result = result.execution_result

        if _is_ambiguous_confirmation(execution_result):
            return (
                "No he entendido la confirmacion. Responde si/s para continuar "
                "o no/n para cancelar."
            )

        pending = _pending_tool_result(execution_result)
        action = _describe_pending_action(pending)
        return f"{action}\nDeseas continuar? [s/N]"


def _missing_information_message(
    fields: tuple[str, ...],
) -> str:
    if not fields:
        return "Necesito mas informacion para poder ejecutar esa accion."

    return "Necesito que indiques: " + ", ".join(fields) + "."


def _ambiguous_information_message(
    fields: tuple[str, ...],
) -> str:
    if not fields:
        return "La peticion es ambigua. Aclara los datos antes de ejecutar."

    return "Necesito que aclares: " + ", ".join(fields) + "."


def _validation_failed_message(
    errors: tuple[str, ...],
) -> str:
    if not errors:
        return "La peticion no ha pasado la validacion."

    return "La peticion no ha pasado la validacion: " + "; ".join(errors)


def _pending_tool_result(
    execution_result: ToolRunResult | ToolChainResult | None,
) -> ToolRunResult | None:
    if isinstance(execution_result, ToolRunResult):
        return execution_result

    if isinstance(execution_result, ToolChainResult) and execution_result.steps:
        return execution_result.steps[-1].result

    return None


def _is_ambiguous_confirmation(
    execution_result: ToolRunResult | ToolChainResult | None,
) -> bool:
    if isinstance(execution_result, ToolRunResult):
        return execution_result.status == "invalid_confirmation"

    if isinstance(execution_result, ToolChainResult):
        return execution_result.status == "invalid_confirmation"

    return False


def _describe_pending_action(
    result: ToolRunResult | None,
) -> str:
    if result is None:
        return "Esta operacion requiere confirmacion."

    path = _argument(result, "path")
    if path:
        return f"Esta operacion requiere confirmacion: {_tool_label(result.tool_name)} sobre {path}."

    return f"Esta operacion requiere confirmacion: {_tool_label(result.tool_name)}."


def _tool_label(
    tool_name: str | None,
) -> str:
    labels = {
        "read_file": "leer archivo",
        "write_file": "escribir archivo",
        "list_directory": "listar directorio",
        "project_tree": "listar arbol del proyecto",
    }

    if tool_name is None:
        return "accion"

    return labels.get(tool_name, tool_name.replace("_", "."))


def _argument(
    result: ToolRunResult,
    name: str,
) -> Any:
    if result.validated_arguments and name in result.validated_arguments:
        return result.validated_arguments[name]

    if result.original_arguments and name in result.original_arguments:
        return result.original_arguments[name]

    return None


def _format_items(
    items: list[Any] | tuple[Any, ...],
) -> str:
    if not items:
        return "No hay elementos."

    return "\n".join(f"- {item}" for item in items)
