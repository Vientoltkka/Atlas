"""Coordinate execution decisions, proposals and tool runners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from tools.execution_decision import (
    ExecutionDecision,
    ExecutionDecisionEngine,
    ExecutionMode,
)
from tools.intent_selector import ToolIntent
from tools.single_tool_runner import SingleToolRunner, ToolRunResult
from tools.tool_chain_proposal_builder import (
    StructuredToolChainProposal,
    ToolChainProposalBuilder,
    ToolChainProposalError,
    ToolChainProposalStatus,
)
from tools.tool_chain_runner import ToolChainResult, ToolChainRunner
from tools.tool_proposal_builder import (
    StructuredToolProposal,
    ToolProposalBuilder,
    ToolProposalError,
    ToolProposalStatus,
)


class ExecutionCoordinationStatus(str, Enum):
    """Uniform status for request execution coordination."""

    DIRECT_RESPONSE_REQUIRED = "DIRECT_RESPONSE_REQUIRED"
    INFORMATION_REQUIRED = "INFORMATION_REQUIRED"
    AMBIGUOUS_REQUEST = "AMBIGUOUS_REQUEST"
    UNSUPPORTED = "UNSUPPORTED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ExecutionCoordinationResult:
    """Uniform result for one coordinated execution request."""

    status: ExecutionCoordinationStatus
    mode: ExecutionMode
    decision: ExecutionDecision
    proposal: StructuredToolProposal | StructuredToolChainProposal | None
    execution_result: ToolRunResult | ToolChainResult | None
    message: str
    missing_information: tuple[str, ...] = ()
    ambiguous_information: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()
    confirmation_id: str | None = None
    executed: bool = False


_ConfirmationOwner = Literal["single", "chain"]


class ExecutionCoordinator:
    """Coordinate decision, proposal conversion and existing tool runners."""

    def __init__(
        self,
        decision_engine: ExecutionDecisionEngine,
        tool_proposal_builder: ToolProposalBuilder,
        chain_proposal_builder: ToolChainProposalBuilder,
        single_tool_runner: SingleToolRunner,
        chain_runner: ToolChainRunner,
    ) -> None:
        self._decision_engine = decision_engine
        self._tool_proposal_builder = tool_proposal_builder
        self._chain_proposal_builder = chain_proposal_builder
        self._single_tool_runner = single_tool_runner
        self._chain_runner = chain_runner
        self._pending_confirmations: dict[str, _ConfirmationOwner] = {}

    def execute(
        self,
        prompt: str,
    ) -> ExecutionCoordinationResult:
        """Coordinate one user request without inventing tools or arguments."""
        decision = self._decision_engine.decide(prompt)
        return self.execute_with_decision(prompt, decision)

    def execute_with_decision(
        self,
        prompt: str,
        decision: ExecutionDecision,
    ) -> ExecutionCoordinationResult:
        """Coordinate one request using an already established decision."""
        if decision.mode is ExecutionMode.DIRECT_RESPONSE:
            if _is_missing_tool_capability(decision):
                return ExecutionCoordinationResult(
                    status=ExecutionCoordinationStatus.UNSUPPORTED,
                    mode=decision.mode,
                    decision=decision,
                    proposal=None,
                    execution_result=None,
                    message=decision.reason,
                    executed=False,
                )

            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED,
                mode=decision.mode,
                decision=decision,
                proposal=None,
                execution_result=None,
                message=decision.reason,
                executed=False,
            )

        if decision.mode is ExecutionMode.SINGLE_TOOL:
            return self._execute_single(prompt, decision)

        if decision.mode is ExecutionMode.TOOL_CHAIN:
            return self._execute_chain(prompt, decision)

        return ExecutionCoordinationResult(
            status=ExecutionCoordinationStatus.UNSUPPORTED,
            mode=decision.mode,
            decision=decision,
            proposal=None,
            execution_result=None,
            message=f"Unsupported execution mode: {decision.mode.value}.",
            executed=False,
        )

    def confirm(
        self,
        confirmation_id: str,
        response: str,
    ) -> ExecutionCoordinationResult:
        """Delegate confirmation to the runner that owns the pending action."""
        owner = self._pending_confirmations.get(confirmation_id)
        decision = _confirmation_decision()

        if owner is None:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.FAILED,
                mode=ExecutionMode.DIRECT_RESPONSE,
                decision=decision,
                proposal=None,
                execution_result=None,
                message="confirmation id is not pending",
                confirmation_id=confirmation_id,
                executed=False,
            )

        if owner == "single":
            outcome = self._single_tool_runner.confirm(confirmation_id, response)
            return self._single_result(
                decision,
                proposal=None,
                outcome=outcome,
                pending_owner="single",
            )

        outcome = self._chain_runner.confirm(confirmation_id, response)
        return self._chain_result(
            decision,
            proposal=None,
            outcome=outcome,
            pending_owner="chain",
        )

    def pending_confirmation_owner(
        self,
        confirmation_id: str,
    ) -> _ConfirmationOwner | None:
        """Return the owner of one pending confirmation id."""
        return self._pending_confirmations.get(confirmation_id)

    def cancel_pending_confirmation(
        self,
        confirmation_id: str,
    ) -> None:
        """Cancel a pending confirmation id without executing any runner."""
        owner = self._pending_confirmations.pop(confirmation_id, None)
        if owner == "single":
            self._single_tool_runner.cancel_pending_confirmation(confirmation_id)
        elif owner == "chain":
            self._chain_runner.cancel_pending_chain(confirmation_id)

    def reissue_single_confirmation(
        self,
        confirmation_id: str,
        arguments: dict[str, object],
    ) -> ExecutionCoordinationResult:
        """Revalidate a modified single-tool confirmation and issue a new id."""
        owner = self._pending_confirmations.get(confirmation_id)
        decision = _confirmation_decision()
        if owner != "single":
            return _confirmation_not_pending_result(confirmation_id, decision)

        outcome = self._single_tool_runner.reissue_pending_confirmation(
            confirmation_id,
            arguments,
        )
        self._pending_confirmations.pop(confirmation_id, None)
        return self._single_result(
            decision,
            proposal=None,
            outcome=outcome,
            pending_owner="single",
        )

    def reissue_chain_confirmation(
        self,
        confirmation_id: str,
        step_arguments: dict[str, object],
        resolved_arguments: dict[str, object],
    ) -> ExecutionCoordinationResult:
        """Revalidate a modified pending chain step and issue a new id."""
        owner = self._pending_confirmations.get(confirmation_id)
        decision = _confirmation_decision()
        if owner != "chain":
            return _confirmation_not_pending_result(confirmation_id, decision)

        outcome = self._chain_runner.reissue_pending_step(
            confirmation_id,
            step_arguments,
            resolved_arguments,
        )
        self._pending_confirmations.pop(confirmation_id, None)
        return self._chain_result(
            decision,
            proposal=None,
            outcome=outcome,
            pending_owner="chain",
        )

    def _execute_single(
        self,
        prompt: str,
        decision: ExecutionDecision,
    ) -> ExecutionCoordinationResult:
        proposal = self._tool_proposal_builder.build(prompt, decision)

        if proposal.status is ToolProposalStatus.INCOMPLETE:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.INFORMATION_REQUIRED,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message="Tool request is missing required information.",
                missing_information=proposal.missing_arguments,
                validation_errors=proposal.validation_errors,
                executed=False,
            )

        if proposal.status is ToolProposalStatus.AMBIGUOUS:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.AMBIGUOUS_REQUEST,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message="Tool request contains ambiguous information.",
                missing_information=proposal.missing_arguments,
                ambiguous_information=proposal.ambiguous_arguments,
                executed=False,
            )

        if proposal.status is ToolProposalStatus.UNSUPPORTED:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.UNSUPPORTED,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message=proposal.reason,
                validation_errors=proposal.validation_errors,
                executed=False,
            )

        try:
            intent = self._tool_proposal_builder.to_tool_intent(proposal)
        except ToolProposalError as error:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.VALIDATION_FAILED,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message=str(error),
                validation_errors=(str(error),),
                executed=False,
            )

        outcome = self._single_tool_runner.run(intent)
        return self._single_result(decision, proposal, outcome, "single")

    def _execute_chain(
        self,
        prompt: str,
        decision: ExecutionDecision,
    ) -> ExecutionCoordinationResult:
        proposal = self._chain_proposal_builder.build(prompt, decision)

        if proposal.status is ToolChainProposalStatus.INCOMPLETE:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.INFORMATION_REQUIRED,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message="Tool chain request is missing required information.",
                missing_information=proposal.missing_information,
                validation_errors=proposal.validation_errors,
                executed=False,
            )

        if proposal.status is ToolChainProposalStatus.AMBIGUOUS:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.AMBIGUOUS_REQUEST,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message="Tool chain request contains ambiguous information.",
                missing_information=proposal.missing_information,
                ambiguous_information=proposal.ambiguous_information,
                validation_errors=proposal.validation_errors,
                executed=False,
            )

        if proposal.status is ToolChainProposalStatus.UNSUPPORTED:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.UNSUPPORTED,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message=proposal.reason,
                missing_information=proposal.missing_information,
                ambiguous_information=proposal.ambiguous_information,
                validation_errors=proposal.validation_errors,
                executed=False,
            )

        try:
            steps = self._chain_proposal_builder.to_tool_chain_definition(proposal)
        except ToolChainProposalError as error:
            return ExecutionCoordinationResult(
                status=ExecutionCoordinationStatus.VALIDATION_FAILED,
                mode=decision.mode,
                decision=decision,
                proposal=proposal,
                execution_result=None,
                message=str(error),
                validation_errors=(str(error),),
                executed=False,
            )

        outcome = self._chain_runner.run(steps)
        return self._chain_result(decision, proposal, outcome, "chain")

    def _single_result(
        self,
        decision: ExecutionDecision,
        proposal: StructuredToolProposal | None,
        outcome: ToolRunResult,
        pending_owner: _ConfirmationOwner,
    ) -> ExecutionCoordinationResult:
        status = _status_from_runner_status(outcome.status, outcome.success)
        confirmation_id = outcome.confirmation_id

        self._update_pending_confirmation(status, confirmation_id, pending_owner)

        return ExecutionCoordinationResult(
            status=status,
            mode=decision.mode,
            decision=decision,
            proposal=proposal,
            execution_result=outcome,
            message=outcome.error_message or outcome.status,
            validation_errors=_runner_validation_errors(outcome),
            confirmation_id=confirmation_id,
            executed=outcome.executed,
        )

    def _chain_result(
        self,
        decision: ExecutionDecision,
        proposal: StructuredToolChainProposal | None,
        outcome: ToolChainResult,
        pending_owner: _ConfirmationOwner,
    ) -> ExecutionCoordinationResult:
        status = _status_from_runner_status(outcome.status, outcome.success)
        confirmation_id = outcome.confirmation_id

        self._update_pending_confirmation(status, confirmation_id, pending_owner)

        return ExecutionCoordinationResult(
            status=status,
            mode=decision.mode,
            decision=decision,
            proposal=proposal,
            execution_result=outcome,
            message=outcome.error_message or outcome.status,
            validation_errors=_chain_validation_errors(outcome),
            confirmation_id=confirmation_id,
            executed=outcome.execution_count > 0,
        )

    def _update_pending_confirmation(
        self,
        status: ExecutionCoordinationStatus,
        confirmation_id: str | None,
        owner: _ConfirmationOwner,
    ) -> None:
        if confirmation_id is None:
            return

        if status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED:
            self._pending_confirmations[confirmation_id] = owner
            return

        self._pending_confirmations.pop(confirmation_id, None)


def _status_from_runner_status(
    status: str,
    success: bool,
) -> ExecutionCoordinationStatus:
    if success:
        return ExecutionCoordinationStatus.EXECUTED
    if status == "confirmation_required":
        return ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    if status == "invalid_confirmation":
        return ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    if status == "cancelled":
        return ExecutionCoordinationStatus.CANCELLED
    if status in {
        "missing_argument",
        "invalid_argument",
        "invalid_argument_type",
        "unexpected_argument",
        "none_not_allowed",
        "schema_not_registered",
    }:
        return ExecutionCoordinationStatus.VALIDATION_FAILED
    return ExecutionCoordinationStatus.FAILED


def _runner_validation_errors(
    outcome: ToolRunResult,
) -> tuple[str, ...]:
    if outcome.error_field is None or outcome.error_message is None:
        return ()
    return (f"{outcome.error_field}: {outcome.error_message}",)


def _chain_validation_errors(
    outcome: ToolChainResult,
) -> tuple[str, ...]:
    if not outcome.error_message:
        return ()
    if outcome.error_code in {
        "missing_argument",
        "invalid_argument",
        "invalid_argument_type",
        "unexpected_argument",
        "none_not_allowed",
        "reference_not_found",
        "reference_field_not_found",
    }:
        return (outcome.error_message,)
    return ()


def _is_missing_tool_capability(decision: ExecutionDecision) -> bool:
    return decision.reason.startswith("No registered tool capability")


def _confirmation_decision() -> ExecutionDecision:
    return ExecutionDecision(
        mode=ExecutionMode.DIRECT_RESPONSE,
        reason="Confirmation request.",
        confidence=1.0,
    )


def _confirmation_not_pending_result(
    confirmation_id: str,
    decision: ExecutionDecision,
) -> ExecutionCoordinationResult:
    return ExecutionCoordinationResult(
        status=ExecutionCoordinationStatus.FAILED,
        mode=ExecutionMode.DIRECT_RESPONSE,
        decision=decision,
        proposal=None,
        execution_result=None,
        message="confirmation id is not pending",
        confirmation_id=confirmation_id,
        executed=False,
    )
