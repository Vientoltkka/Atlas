from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from core.execution_authorization import (
    AuthorizationValidationStatus,
    ConfirmationScope,
    DispatchStatus,
    ExecutionAuthorizationDecision,
    ExecutionAuthorizationGate,
    ExecutionAuthorizationRequest,
    ExecutionConfirmationReference,
    ExecutionDispatcher,
    authorization_with_dispatch,
    build_confirmation_references,
)
from core.execution_plan_executor import ExecutionPlanExecutor
from core.execution_plan_validator import (
    ExecutionPlanValidator,
    plan_signature,
)
from core.execution_retry import RetryPolicy, RetryStrategy
from core.execution_strategy import (
    ExecutionStrategySelectionResult,
    ExecutionStrategySelector,
    ExecutionStrategyType,
    FailureBehavior,
    StrategyValidationStatus,
    build_strategy_request,
)
from core.execution_report import ExecutionReportGenerator
from core.execution_session_persistence import (
    ExecutionSessionSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)
from core.execution_supervisor import ExecutionSupervisor
from core.planner import ExecutionPlan, ExecutionStep, PlanGenerationResult
from core.structured_execution import StructuredExecutionCoordinator
from tools.base_tool import BaseTool
from tools.registry import ToolRegistry


NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)


def _plan(
    *,
    criticality: int = 0,
    confirmation: bool = False,
    retry_policy: RetryPolicy | None = None,
    status: str = "planned",
    goal: str = "Preparar informe",
) -> ExecutionPlan:
    return ExecutionPlan(
        goal=goal,
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "Preparar",
                "direct_response",
                retry_policy=retry_policy,
                side_effect_free=not confirmation,
                criticality=criticality,
            ),
        ),
        estimated_steps=1,
        required_tools=("direct_response",),
        detected_risks=(),
        requires_confirmation=confirmation,
        status=status,
    )


def _selection(
    plan: ExecutionPlan,
    *,
    supervisor: bool = True,
    replanner: bool = False,
) -> ExecutionStrategySelectionResult:
    validation = ExecutionPlanValidator().validate(plan)
    return ExecutionStrategySelector().select(
        build_strategy_request(
            plan,
            validation,
            supervisor_available=supervisor,
            replanner_available=replanner,
        )
    )


def _confirmation(
    plan: ExecutionPlan,
    selection: ExecutionStrategySelectionResult,
    *,
    confirmation_id: str = "confirmation.current",
    step_id: str = "step_1",
    granted: bool = False,
    revoked: bool = False,
    expired: bool = False,
    consumed: bool = False,
) -> ExecutionConfirmationReference:
    signature = plan_signature(plan)
    return ExecutionConfirmationReference(
        confirmation_id=confirmation_id,
        plan_id=selection.strategy.plan_id,
        plan_signature=signature,
        step_id=step_id,
        tool="direct_response",
        scope=ConfirmationScope.STEP,
        issued_at=NOW - timedelta(minutes=5),
        expires_at=(
            NOW - timedelta(seconds=1)
            if expired
            else NOW + timedelta(minutes=5)
        ),
        granted=granted,
        revoked=revoked,
        consumed=consumed,
    )


def _request(
    plan: ExecutionPlan,
    selection: ExecutionStrategySelectionResult,
    *,
    required=(),
    granted=(),
    executor: bool = True,
    supervisor: bool = True,
    replanner: bool = False,
    plan_state: str | None = None,
    session_state: str | None = None,
) -> ExecutionAuthorizationRequest:
    return ExecutionAuthorizationRequest(
        plan=plan,
        plan_validation=ExecutionPlanValidator().validate(plan),
        strategy_selection=selection,
        plan_state=plan.status if plan_state is None else plan_state,
        session_state=session_state,
        required_confirmations=tuple(required),
        granted_confirmations=tuple(granted),
        protected_step_ids=(
            selection.strategy.configuration.confirmation_step_ids
        ),
        required_capabilities=plan.required_tools,
        executor_available=executor,
        supervisor_available=supervisor,
        replanner_available=replanner,
        source="test",
    )


def _gate(dispatcher: ExecutionDispatcher | None = None):
    active_dispatcher = dispatcher or ExecutionDispatcher(clock=lambda: NOW)
    return (
        ExecutionAuthorizationGate(
            clock=lambda: NOW,
            already_dispatched=active_dispatcher.has_dispatched,
            confirmation_consumed=active_dispatcher.confirmation_consumed,
        ),
        active_dispatcher,
    )


def _manual_selection(
    selection: ExecutionStrategySelectionResult,
) -> ExecutionStrategySelectionResult:
    config = replace(
        selection.strategy.configuration,
        effective_max_retry_attempts=1,
        failure_behavior=FailureBehavior.BLOCK_EXECUTION,
        allow_replanning=False,
        max_replans=0,
        execution_allowed=False,
    )
    strategy = replace(
        selection.strategy,
        type=ExecutionStrategyType.MANUAL_REVIEW_REQUIRED,
        configuration=config,
        validation_status=StrategyValidationStatus.BLOCKED,
    )
    validation = replace(
        selection.validation,
        status=StrategyValidationStatus.BLOCKED,
        is_valid=True,
        executable=False,
        errors=(),
    )
    return replace(
        selection,
        strategy=strategy,
        validation=validation,
    )


def test_standard_is_authorized() -> None:
    plan = _plan()
    result = _gate()[0].authorize(_request(plan, _selection(plan)))

    assert result.decision is ExecutionAuthorizationDecision.AUTHORIZED
    assert result.dispatch_allowed is True
    assert result.plan_signature == plan_signature(plan)


def test_supervised_requires_available_supervisor() -> None:
    plan = _plan(criticality=2)
    selection = _selection(plan)
    gate, _ = _gate()

    authorized = gate.authorize(_request(plan, selection))
    blocked = gate.authorize(
        _request(plan, selection, supervisor=False)
    )

    assert selection.strategy.type is ExecutionStrategyType.SUPERVISED
    assert authorized.decision is ExecutionAuthorizationDecision.AUTHORIZED
    assert blocked.decision is ExecutionAuthorizationDecision.BLOCKED
    assert blocked.missing_components == ("ExecutionSupervisor",)


def test_confirmation_required_remains_pending_without_confirmation() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)

    result = _gate()[0].authorize(
        _request(plan, selection, required=(required,))
    )

    assert result.decision is ExecutionAuthorizationDecision.CONFIRMATION_PENDING
    assert result.dispatch_allowed is False
    assert result.pending_confirmation_ids == ("confirmation.current",)


def test_valid_scoped_confirmation_authorizes() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)

    result = _gate()[0].authorize(
        _request(
            plan,
            selection,
            required=(required,),
            granted=(required.granted_copy(),),
        )
    )

    assert result.decision is ExecutionAuthorizationDecision.AUTHORIZED
    assert result.trace["satisfied_confirmation_ids"] == (
        "confirmation.current",
    )


def test_confirmation_from_other_plan_is_rejected() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)
    foreign = replace(required.granted_copy(), plan_id="foreign.plan")

    result = _gate()[0].authorize(
        _request(plan, selection, required=(required,), granted=(foreign,))
    )

    assert result.decision is ExecutionAuthorizationDecision.REJECTED
    assert "another plan" in result.reason


def test_confirmation_from_other_step_is_rejected() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)
    foreign = replace(required.granted_copy(), step_id="step_2")

    result = _gate()[0].authorize(
        _request(plan, selection, required=(required,), granted=(foreign,))
    )

    assert result.decision is ExecutionAuthorizationDecision.REJECTED
    assert "another plan or step" in result.reason


def test_expired_revoked_or_consumed_confirmation_is_rejected() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    gate, _ = _gate()
    required = _confirmation(plan, selection)

    for granted, message in (
        (_confirmation(plan, selection, granted=True, expired=True), "expired"),
        (_confirmation(plan, selection, granted=True, revoked=True), "revoked"),
        (_confirmation(plan, selection, granted=True, consumed=True), "consumed"),
    ):
        result = gate.authorize(
            _request(
                plan,
                selection,
                required=(required,),
                granted=(granted,),
            )
        )
        assert result.decision is ExecutionAuthorizationDecision.REJECTED
        assert message in result.reason.lower()


def test_changed_plan_invalidates_confirmation_and_authorization() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)
    changed = replace(plan, goal="Objetivo cambiado")

    request = replace(
        _request(
            plan,
            selection,
            required=(required,),
            granted=(required.granted_copy(),),
        ),
        plan=changed,
    )
    result = _gate()[0].authorize(request)

    assert result.decision is ExecutionAuthorizationDecision.REJECTED
    assert "signature" in result.reason.lower()


def test_manual_review_never_authorizes_automatic_dispatch() -> None:
    plan = _plan()
    selection = _manual_selection(_selection(plan))

    result = _gate()[0].authorize(_request(plan, selection))

    assert result.decision is ExecutionAuthorizationDecision.MANUAL_REVIEW_PENDING
    assert result.dispatch_allowed is False


def test_invalid_plan_or_strategy_is_rejected() -> None:
    invalid_plan = _plan(status="running")
    invalid_selection = _selection(invalid_plan)
    invalid_plan_result = _gate()[0].authorize(
        _request(invalid_plan, invalid_selection)
    )
    plan = _plan()
    selection = _selection(plan)
    invalid_strategy = replace(
        selection,
        validation=replace(
            selection.validation,
            status=StrategyValidationStatus.INVALID,
            is_valid=False,
            executable=False,
            errors=("invalid",),
        ),
    )
    invalid_strategy_result = _gate()[0].authorize(
        _request(plan, invalid_strategy)
    )

    assert invalid_plan_result.decision is ExecutionAuthorizationDecision.REJECTED
    assert invalid_strategy_result.decision is ExecutionAuthorizationDecision.REJECTED


def test_active_and_terminal_states_are_rejected() -> None:
    plan = _plan()
    selection = _selection(plan)
    gate, _ = _gate()

    for state in ("executing", "completed", "cancelled", "failed"):
        result = gate.authorize(
            _request(plan, selection, session_state=state)
        )
        assert result.decision is ExecutionAuthorizationDecision.REJECTED


def test_dispatch_is_atomic_and_idempotent() -> None:
    plan = _plan()
    gate, dispatcher = _gate()
    authorization = gate.authorize(_request(plan, _selection(plan)))
    calls = []

    first = dispatcher.dispatch(
        authorization,
        lambda plan, validation, strategy: (
            calls.append(plan.goal) or "session.123"
        ),
    )
    second = dispatcher.dispatch(
        authorization,
        lambda plan, validation, strategy: calls.append("duplicate"),
    )
    repeated_authorization = gate.authorize(
        _request(plan, _selection(plan))
    )

    assert first.status is DispatchStatus.DISPATCHED
    assert first.session_id == "session.123"
    assert second.status is DispatchStatus.ALREADY_DISPATCHED
    assert repeated_authorization.decision is (
        ExecutionAuthorizationDecision.ALREADY_DISPATCHED
    )
    assert calls == ["Preparar informe"]


def test_confirmation_cannot_be_reused_with_another_request_source() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)
    gate, dispatcher = _gate()
    first_request = _request(
        plan,
        selection,
        required=(required,),
        granted=(required.granted_copy(),),
    )
    first = gate.authorize(first_request)
    dispatcher.dispatch(first, lambda *args: "session.confirmed")

    reused = gate.authorize(
        replace(first_request, source="another-request-source")
    )

    assert reused.decision is ExecutionAuthorizationDecision.REJECTED
    assert "already consumed" in reused.reason


def test_pre_dispatch_failure_does_not_consume_or_call_delivery() -> None:
    plan = _plan()
    authorization = _gate()[0].authorize(
        _request(plan, _manual_selection(_selection(plan)))
    )
    dispatcher = ExecutionDispatcher(clock=lambda: NOW)
    calls = []

    result = dispatcher.dispatch(
        authorization,
        lambda *args: calls.append(args),
    )

    assert result.status is DispatchStatus.BLOCKED
    assert result.consumed is False
    assert dispatcher.has_dispatched(authorization.authorization_id) is False
    assert calls == []


def test_failure_after_delivery_is_recorded_and_not_repeated() -> None:
    plan = _plan()
    authorization = _gate()[0].authorize(
        _request(plan, _selection(plan))
    )
    dispatcher = ExecutionDispatcher(clock=lambda: NOW)
    calls = []

    def fail(*args):
        calls.append("delivered")
        raise RuntimeError("api_token=secret")

    failed = dispatcher.dispatch(authorization, fail)
    repeated = dispatcher.dispatch(authorization, fail)

    assert failed.status is DispatchStatus.FAILED_AFTER_DISPATCH
    assert failed.consumed is True
    assert failed.error == "[redacted]"
    assert repeated.status is DispatchStatus.ALREADY_DISPATCHED
    assert calls == ["delivered"]


def test_plan_invariants_remain_unchanged_through_dispatch() -> None:
    plan = _plan(criticality=3)
    before = plan_signature(plan)
    selection = _selection(plan)
    authorization = _gate()[0].authorize(_request(plan, selection))
    observed = []

    ExecutionDispatcher(clock=lambda: NOW).dispatch(
        authorization,
        lambda delivered, validation, strategy: observed.append(
            (
                delivered.goal,
                delivered.required_tools,
                tuple(step.id for step in delivered.ordered_steps),
                tuple(step.tool for step in delivered.ordered_steps),
                tuple(step.criticality for step in delivered.ordered_steps),
            )
        )
        or "session.invariant",
    )

    assert before == plan_signature(plan)
    assert observed == [
        (
            "Preparar informe",
            ("direct_response",),
            ("step_1",),
            ("direct_response",),
            (3,),
        )
    ]


class _FailingTool(BaseTool):
    name = "transient_tool"
    description = "Always fails."
    requires_confirmation = False

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, context):
        self.calls += 1
        raise TimeoutError("transient")


def test_strategy_authorization_dispatcher_and_executor_preserve_no_retry() -> None:
    tool = _FailingTool()
    registry = ToolRegistry()
    registry.register(tool)
    no_retry = RetryPolicy(
        max_attempts=1,
        strategy=RetryStrategy.NO_RETRY,
    )
    plan = ExecutionPlan(
        goal="Do not retry",
        ordered_steps=(
            ExecutionStep(
                "step_1",
                "Fail once",
                "transient_tool",
                retry_policy=no_retry,
                side_effect_free=True,
            ),
        ),
        estimated_steps=1,
        required_tools=("transient_tool",),
        detected_risks=(),
        requires_confirmation=False,
    )
    validator = ExecutionPlanValidator(registry)
    validation = validator.validate(plan)
    before = plan_signature(plan)
    selection = ExecutionStrategySelector().select(
        build_strategy_request(plan, validation)
    )
    gate, dispatcher = _gate()
    authorization = gate.authorize(
        replace(
            _request(plan, selection),
            plan_validation=validation,
        )
    )
    executor = ExecutionPlanExecutor(registry)

    dispatch = dispatcher.dispatch(
        authorization,
        lambda delivered, validated, selected: executor.execute(
            delivered,
            validated,
            operational_config=selected.strategy.configuration,
        ),
    )

    assert selection.strategy.configuration.effective_max_retry_attempts == 1
    assert authorization.decision is ExecutionAuthorizationDecision.AUTHORIZED
    assert dispatch.status is DispatchStatus.DISPATCHED
    assert tool.calls == 1
    assert plan.ordered_steps[0].retry_policy.strategy is RetryStrategy.NO_RETRY
    assert plan.ordered_steps[0].retry_policy.max_attempts == 1
    assert plan_signature(plan) == before


def test_confirmation_reference_builder_is_scoped_and_bounded() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)

    references = build_confirmation_references(
        plan,
        selection,
        confirmation_id="confirmation.current",
        issued_at=NOW,
        ttl_seconds=60,
    )

    assert len(references) == 1
    assert references[0].step_id == "step_1"
    assert references[0].tool == "direct_response"
    assert references[0].expires_at == NOW + timedelta(seconds=60)


def test_authorization_snapshot_contains_no_confirmation_content() -> None:
    plan = _plan(confirmation=True)
    selection = _selection(plan)
    required = _confirmation(plan, selection)
    authorization = _gate()[0].authorize(
        _request(
            plan,
            selection,
            required=(required,),
            granted=(required.granted_copy(),),
        )
    )
    dispatch = ExecutionDispatcher(clock=lambda: NOW).dispatch(
        authorization,
        lambda *args: "session.safe",
    )

    snapshot = authorization_with_dispatch(authorization, dispatch)

    assert snapshot["decision"] == "AUTHORIZED"
    assert snapshot["satisfied_confirmation_ids"] == [
        "confirmation.current"
    ]
    assert "content" not in snapshot
    assert snapshot["session_id"] == "session.safe"
    assert authorization.validation_status is AuthorizationValidationStatus.VALID


def test_main_coordinator_flow_authorizes_and_dispatches_before_executor() -> None:
    plan = _plan()
    registry = ToolRegistry()

    class _Planner:
        def generate_execution_plan(self, objective, **kwargs):
            return PlanGenerationResult(success=True, plan=plan)

    coordinator = StructuredExecutionCoordinator(
        planner=_Planner(),
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(registry),
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    response = coordinator.handle("Preparar informe")

    assert response.status == "completed"
    assert response.authorization_result is not None
    assert response.authorization_result.decision is (
        ExecutionAuthorizationDecision.AUTHORIZED
    )
    assert response.dispatch_result is not None
    assert response.dispatch_result.status is DispatchStatus.DISPATCHED
    assert response.execution_result is not None
    assert response.operational_report is not None
    assert response.operational_report.authorization_status == "AUTHORIZED"
    assert response.operational_report.authorization_ready is False
    assert response.operational_report.dispatch_completed is True


def test_main_confirmation_flow_reuses_pending_confirmation_and_authorizes() -> None:
    plan = _plan(confirmation=True)

    class _Planner:
        def generate_execution_plan(self, objective, **kwargs):
            return PlanGenerationResult(success=True, plan=plan)

    coordinator = StructuredExecutionCoordinator(
        planner=_Planner(),
        validator=ExecutionPlanValidator(),
        executor=ExecutionPlanExecutor(ToolRegistry()),
        execution_strategy_selector=ExecutionStrategySelector(),
    )

    pending = coordinator.handle("Preparar informe")
    confirmed = coordinator.confirm(pending.confirmation_token or "")

    assert pending.status == "confirmation_required"
    assert pending.authorization_result is not None
    assert pending.authorization_result.decision is (
        ExecutionAuthorizationDecision.CONFIRMATION_PENDING
    )
    assert confirmed.status == "completed"
    assert confirmed.authorization_result is not None
    assert confirmed.authorization_result.decision is (
        ExecutionAuthorizationDecision.AUTHORIZED
    )
    assert confirmed.dispatch_result is not None
    assert confirmed.dispatch_result.consumed is True


def test_authorization_persistence_and_legacy_report_compatibility() -> None:
    plan = _plan()
    selection = _selection(plan)
    authorization = _gate()[0].authorize(_request(plan, selection))
    dispatch = ExecutionDispatcher(clock=lambda: NOW).dispatch(
        authorization,
        lambda *args: "execution.session.000001",
    )
    supervisor = ExecutionSupervisor(clock=lambda: NOW)
    session = supervisor.start(
        plan,
        execution_strategy=selection.persisted_snapshot(),
        execution_authorization=authorization_with_dispatch(
            authorization,
            dispatch,
        ),
    )
    snapshot = ExecutionSessionSnapshot.from_session(session)
    payload = snapshot_to_dict(snapshot)

    restored = snapshot_from_dict(payload)
    report = ExecutionReportGenerator().generate(
        restored.to_session(),
        supervisor.get_summary(session.session_id),
    )

    assert restored.execution_authorization["decision"] == "AUTHORIZED"
    assert report.authorization_status == "AUTHORIZED"
    assert report.dispatch_completed is True
    assert report.authorization_session_id == "execution.session.000001"

    payload.pop("execution_authorization")
    legacy = snapshot_from_dict(payload)
    legacy_report = ExecutionReportGenerator().generate(
        legacy.to_session(),
        supervisor.get_summary(session.session_id),
    )
    assert legacy.execution_authorization is None
    assert legacy_report.authorization_status == "LEGACY_NOT_RECORDED"
    assert legacy_report.dispatch_completed is False


def test_source_and_dispatch_errors_are_sanitized() -> None:
    plan = _plan()
    selection = _selection(plan)
    request = replace(
        _request(plan, selection),
        source="api_token=secret",
    )
    authorization = _gate()[0].authorize(request)

    assert authorization.trace["source"] == "[redacted]"
