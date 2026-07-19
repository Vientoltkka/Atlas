"""Conversational adapter for structured execution coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.execution_coordinator import (
    ExecutionCoordinationResult,
    ExecutionCoordinationStatus,
    ExecutionCoordinator,
)
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.single_tool_runner import ToolRunResult
from tools.tool_chain_runner import ToolChainResult, ToolChainStepResult
from use_cases.execution_clarification import (
    ClarificationQuestionPresenter,
    ClarificationResolver,
    PendingClarification,
    is_cancel_clarification,
    merge_fields,
    synthetic_result,
)
from use_cases.pending_confirmation import (
    PendingConfirmationContext,
    PendingConfirmationInputType,
    PendingConfirmationPresenter,
    PendingConfirmationResolver,
    context_with_new_result,
)


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
        self._question_presenter = ClarificationQuestionPresenter()
        self._clarification_resolver = ClarificationResolver()
        self._pending_confirmation_resolver = PendingConfirmationResolver()
        self._pending_confirmation_presenter = PendingConfirmationPresenter()
        self._pending_confirmation_id: str | None = None
        self._pending_confirmation_context: PendingConfirmationContext | None = None
        self._pending_clarification: PendingClarification | None = None
        self._last_result: ExecutionCoordinationResult | None = None

    @property
    def pending_confirmation_id(self) -> str | None:
        """Return the active confirmation id for debugging."""
        return self._pending_confirmation_id

    @property
    def last_result(self) -> ExecutionCoordinationResult | None:
        """Return the latest structured result for debugging."""
        return self._last_result

    @property
    def pending_clarification(self) -> PendingClarification | None:
        """Return the active clarification request for debugging."""
        return self._pending_clarification

    @property
    def pending_confirmation_context(self) -> PendingConfirmationContext | None:
        """Return the active confirmation context for debugging."""
        return self._pending_confirmation_context

    def handle(
        self,
        prompt: str,
    ) -> ExecutionConversationOutcome:
        """Handle one text turn before the normal conversational fallback."""
        if self._pending_confirmation_id is not None:
            return self._handle_confirmation_response(prompt)

        if self._pending_clarification is not None:
            return self._handle_clarification_response(prompt)

        result = self._coordinator.execute(prompt)
        self._remember(result, original_text=prompt)

        if result.status is ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED:
            return ExecutionConversationOutcome(
                direct_response_required=True,
                text="",
                result=result,
            )

        if result.status in {
            ExecutionCoordinationStatus.INFORMATION_REQUIRED,
            ExecutionCoordinationStatus.AMBIGUOUS_REQUEST,
        }:
            return ExecutionConversationOutcome(
                direct_response_required=False,
                text=self._question_presenter.present(self._pending_clarification),
                result=result,
            )

        return ExecutionConversationOutcome(
            direct_response_required=False,
            text=self._presenter.present(result),
            result=result,
        )

    def _handle_confirmation_response(
        self,
        prompt: str,
    ) -> ExecutionConversationOutcome:
        context = self._pending_confirmation_context
        if context is None:
            raise RuntimeError("No pending confirmation context is available.")

        resolution = self._pending_confirmation_resolver.resolve(prompt, context)

        if resolution.input_type is PendingConfirmationInputType.CONFIRM:
            result = self._coordinator.confirm(context.confirmation_id, prompt)
            self._remember(result)
            return ExecutionConversationOutcome(False, self._presenter.present(result), result)

        if resolution.input_type in {
            PendingConfirmationInputType.REJECT,
            PendingConfirmationInputType.CANCEL,
        }:
            result = self._coordinator.confirm(context.confirmation_id, "no")
            self._remember(result)
            return ExecutionConversationOutcome(False, self._presenter.present(result), result)

        if resolution.input_type is PendingConfirmationInputType.INSPECT:
            result = _pending_confirmation_result(context, "pending operation inspected")
            self._last_result = result
            return ExecutionConversationOutcome(
                False,
                self._pending_confirmation_presenter.describe(context, inspect=True),
                result,
            )

        if resolution.input_type is PendingConfirmationInputType.REPLACE:
            self._coordinator.cancel_pending_confirmation(context.confirmation_id)
            self._clear_pending_confirmation()
            replacement = resolution.replacement_text or prompt
            return self.handle(replacement)

        if resolution.input_type is PendingConfirmationInputType.MODIFY:
            if resolution.blocked_reason:
                result = _pending_confirmation_result(context, resolution.blocked_reason)
                self._last_result = result
                return ExecutionConversationOutcome(False, resolution.blocked_reason, result)

            if not resolution.arguments:
                result = _pending_confirmation_result(context, "modification is incomplete")
                self._last_result = result
                return ExecutionConversationOutcome(
                    False,
                    (
                        "La modificacion no es suficientemente clara. Indica que "
                        "argumento quieres cambiar o confirma/cancela la operacion pendiente."
                    ),
                    result,
                )

            return self._apply_pending_modification(context, dict(resolution.arguments))

        result = _pending_confirmation_result(context, "pending confirmation needs a clear response")
        self._last_result = result
        return ExecutionConversationOutcome(
            False,
            (
                "Hay una confirmacion pendiente. Responde si/s para ejecutar, "
                "no/n para cancelar, pregunta que voy a hacer o indica un cambio claro."
            ),
            result,
        )

    def _apply_pending_modification(
        self,
        context: PendingConfirmationContext,
        arguments: dict[str, Any],
    ) -> ExecutionConversationOutcome:
        pending = _pending_tool_result(context.execution_result)
        if pending is None:
            result = _pending_confirmation_result(context, "pending operation cannot be modified")
            self._last_result = result
            return ExecutionConversationOutcome(False, "No puedo modificar esa operacion pendiente.", result)

        current_arguments = dict(pending.validated_arguments or pending.original_arguments or {})
        current_arguments.update(arguments)

        if context.owner == "single":
            result = self._coordinator.reissue_single_confirmation(
                context.confirmation_id,
                current_arguments,
            )
        else:
            step = _pending_chain_step(context.execution_result)
            step_arguments = dict(step.arguments) if step is not None else {}
            step_arguments.update(arguments)
            result = self._coordinator.reissue_chain_confirmation(
                context.confirmation_id,
                step_arguments,
                current_arguments,
            )

        self._remember(result, original_text=context.original_text)

        if result.status in {
            ExecutionCoordinationStatus.INFORMATION_REQUIRED,
            ExecutionCoordinationStatus.AMBIGUOUS_REQUEST,
        }:
            return ExecutionConversationOutcome(
                False,
                self._question_presenter.present(self._pending_clarification),
                result,
            )

        if result.status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED:
            active_context = self._pending_confirmation_context
            if active_context is not None:
                text = self._pending_confirmation_presenter.describe(
                    active_context,
                    prefix="De acuerdo. Ahora",
                )
            else:
                text = self._presenter.present(result)
            return ExecutionConversationOutcome(False, text, result)

        return ExecutionConversationOutcome(False, self._presenter.present(result), result)

    def _handle_clarification_response(
        self,
        prompt: str,
    ) -> ExecutionConversationOutcome:
        pending = self._pending_clarification
        if pending is None:
            raise RuntimeError("No pending clarification is available.")

        if is_cancel_clarification(prompt):
            result = synthetic_result(
                ExecutionCoordinationStatus.CANCELLED,
                pending.mode,
                "operation cancelled by user",
                pending.proposal,
                candidate_tools=pending.candidate_tools,
            )
            self._remember(result)
            return ExecutionConversationOutcome(False, self._presenter.present(result), result)

        if not prompt.strip():
            result = synthetic_result(
                ExecutionCoordinationStatus.INFORMATION_REQUIRED,
                pending.mode,
                "clarification response is empty",
                pending.proposal,
                candidate_tools=pending.candidate_tools,
                missing_information=pending.missing_information,
                ambiguous_information=pending.ambiguous_information,
            )
            self._remember_clarification_result(result, pending.original_text)
            return ExecutionConversationOutcome(
                False,
                self._question_presenter.present(self._pending_clarification),
                result,
            )

        if self._clarification_resolver.looks_like_new_order(prompt, pending):
            result = synthetic_result(
                ExecutionCoordinationStatus.INFORMATION_REQUIRED,
                pending.mode,
                "new order received while clarification is pending",
                pending.proposal,
                candidate_tools=pending.candidate_tools,
                missing_information=pending.missing_information,
                ambiguous_information=pending.ambiguous_information,
            )
            self._last_result = result
            return ExecutionConversationOutcome(
                False,
                (
                    "Hay una operacion pendiente de aclaracion. "
                    "Responde a esa aclaracion o escribe cancelar para descartarla."
                ),
                result,
            )

        completed_text = self._clarification_resolver.resolve(pending, prompt)
        if completed_text is None:
            result = synthetic_result(
                ExecutionCoordinationStatus.INFORMATION_REQUIRED,
                pending.mode,
                "clarification response did not provide requested fields",
                pending.proposal,
                candidate_tools=pending.candidate_tools,
                missing_information=pending.missing_information,
                ambiguous_information=pending.ambiguous_information,
            )
            self._last_result = result
            return ExecutionConversationOutcome(
                False,
                self._question_presenter.present(pending),
                result,
            )

        result = self._coordinator.execute_with_decision(
            completed_text,
            ExecutionDecision(
                mode=pending.mode,
                reason="Clarified pending operation.",
                confidence=1.0,
                candidate_tools=pending.candidate_tools,
            ),
        )
        self._remember(result, original_text=completed_text)

        if result.status in {
            ExecutionCoordinationStatus.INFORMATION_REQUIRED,
            ExecutionCoordinationStatus.AMBIGUOUS_REQUEST,
        }:
            return ExecutionConversationOutcome(
                False,
                self._question_presenter.present(self._pending_clarification),
                result,
            )

        return ExecutionConversationOutcome(
            direct_response_required=False,
            text=self._presenter.present(result),
            result=result,
        )

    def _remember(
        self,
        result: ExecutionCoordinationResult,
        original_text: str | None = None,
    ) -> None:
        self._last_result = result

        if result.status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED:
            self._pending_clarification = None
            self._pending_confirmation_id = result.confirmation_id
            self._pending_confirmation_context = self._build_pending_confirmation_context(
                result,
                original_text,
            )
            return

        self._pending_confirmation_id = None
        self._pending_confirmation_context = None

        if result.status in {
            ExecutionCoordinationStatus.INFORMATION_REQUIRED,
            ExecutionCoordinationStatus.AMBIGUOUS_REQUEST,
        } and result.proposal is not None:
            self._remember_clarification_result(
                result,
                original_text or result.proposal.source_text,
            )
            return

        self._pending_clarification = None

    def _build_pending_confirmation_context(
        self,
        result: ExecutionCoordinationResult,
        original_text: str | None,
    ) -> PendingConfirmationContext | None:
        if result.confirmation_id is None:
            return None

        owner = self._coordinator.pending_confirmation_owner(result.confirmation_id)
        if owner is None:
            previous = self._pending_confirmation_context
            if previous is not None and previous.confirmation_id == result.confirmation_id:
                return previous
            return None

        previous = self._pending_confirmation_context
        if previous is not None and previous.confirmation_id == result.confirmation_id:
            return context_with_new_result(
                previous,
                result.execution_result or previous.execution_result,
                result.confirmation_id,
            )

        context = PendingConfirmationContext(
            confirmation_id=result.confirmation_id,
            mode=result.mode,
            owner=owner,
            original_text=original_text or (result.proposal.source_text if result.proposal else ""),
            proposal=result.proposal,
            execution_result=result.execution_result,
            action_summary=self._presenter._confirmation_message(result),
        )
        return context

    def _clear_pending_confirmation(self) -> None:
        self._pending_confirmation_id = None
        self._pending_confirmation_context = None
        self._pending_clarification = None

    def _remember_clarification_result(
        self,
        result: ExecutionCoordinationResult,
        original_text: str,
    ) -> None:
        self._last_result = result
        if result.proposal is None:
            return

        requested = merge_fields(
            result.missing_information,
            result.ambiguous_information,
        )
        self._pending_clarification = PendingClarification(
            original_text=original_text,
            mode=result.mode,
            proposal=result.proposal,
            candidate_tools=result.decision.candidate_tools,
            missing_information=result.missing_information,
            ambiguous_information=result.ambiguous_information,
            requested_fields=requested,
        )


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


def _pending_chain_step(
    execution_result: ToolRunResult | ToolChainResult | None,
) -> ToolChainStepResult | None:
    if isinstance(execution_result, ToolChainResult) and execution_result.steps:
        return execution_result.steps[-1]
    return None


def _pending_confirmation_result(
    context: PendingConfirmationContext,
    message: str,
) -> ExecutionCoordinationResult:
    return ExecutionCoordinationResult(
        status=ExecutionCoordinationStatus.CONFIRMATION_REQUIRED,
        mode=context.mode,
        decision=ExecutionDecision(
            mode=context.mode,
            reason=message,
            confidence=1.0,
        ),
        proposal=context.proposal,
        execution_result=context.execution_result,
        message=message,
        confirmation_id=context.confirmation_id,
        executed=False,
    )


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
