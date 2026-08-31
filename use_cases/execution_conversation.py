"""Conversational adapter for structured execution coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

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
from use_cases.execution_result_presenter import ExecutionResultPresenter
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
        restore_pending_target: Callable[[int], None] | None = None,
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
        self._restore_pending_target = restore_pending_target
        self._pending_target_handles: dict[str, int] = {}

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

    @property
    def pending_target_handle(self) -> int | None:
        """Return the exact desktop target retained for the pending action."""
        if self._pending_confirmation_id is None:
            return None
        return self._pending_target_handles.get(self._pending_confirmation_id)

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

    def handle_registered_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        original_text: str,
        confirmation_text: str,
        pending_target_handle: int | None = None,
    ) -> ExecutionConversationOutcome:
        """Prepare an exact tool while preserving this controller's confirmation state."""
        result = self._coordinator.execute_registered_tool(tool_name, arguments)
        self._remember(result, original_text=original_text)
        if result.confirmation_id is not None and pending_target_handle is not None:
            self._pending_target_handles[result.confirmation_id] = pending_target_handle
        text = (
            confirmation_text
            if result.status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
            else self._presenter.present(result)
        )
        return ExecutionConversationOutcome(False, text, result)
    def _handle_confirmation_response(
        self,
        prompt: str,
    ) -> ExecutionConversationOutcome:
        context = self._pending_confirmation_context
        if context is None:
            raise RuntimeError("No pending confirmation context is available.")

        resolution = self._pending_confirmation_resolver.resolve(prompt, context)

        if resolution.input_type is PendingConfirmationInputType.CONFIRM:
            target_handle = self._pending_target_handles.pop(
                context.confirmation_id,
                None,
            )
            if target_handle is not None and self._restore_pending_target is not None:
                try:
                    self._restore_pending_target(target_handle)
                except RuntimeError as error:
                    self._coordinator.cancel_pending_confirmation(context.confirmation_id)
                    self._clear_pending_confirmation()
                    result = ExecutionCoordinationResult(
                        status=ExecutionCoordinationStatus.FAILED,
                        mode=context.mode,
                        decision=ExecutionDecision(
                            mode=context.mode,
                            reason="pending desktop target is unavailable",
                            confidence=1.0,
                        ),
                        proposal=context.proposal,
                        execution_result=context.execution_result,
                        message=str(error),
                        confirmation_id=context.confirmation_id,
                        executed=False,
                    )
                    self._last_result = result
                    return ExecutionConversationOutcome(
                        False,
                        "La ventana destino ya no esta disponible. No se ha escrito nada.",
                        result,
                    )
            result = self._coordinator.confirm(context.confirmation_id, prompt)
            self._remember(result)
            return ExecutionConversationOutcome(False, self._presenter.present(result), result)

        if resolution.input_type in {
            PendingConfirmationInputType.REJECT,
            PendingConfirmationInputType.CANCEL,
        }:
            self._pending_target_handles.pop(context.confirmation_id, None)
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
        if self._pending_confirmation_id is not None:
            self._pending_target_handles.pop(self._pending_confirmation_id, None)
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

