from decimal import Decimal
from pathlib import Path

from finance.execution.adapters import InMemoryPaperAdapter
from finance.execution.ledger import ExecutionLedger
from finance.execution.models import ExecutionIntent, ExecutionMode, ExecutionStatus
from finance.execution.policy import RiskPolicyStore
from finance.execution.risk_gate import PortfolioRiskState
from finance.execution.service import ExecutionService
from finance.paper.models import OrderType, Side


def _state(**changes):
    values = dict(
        nav=Decimal("10000"),
        cash=Decimal("10000"),
        positions={},
        prices={"AAPL": Decimal("100")},
        daily_trade_count=0,
        daily_realized_loss=Decimal("0"),
    )
    values.update(changes)
    return PortfolioRiskState(**values)


def _intent(**changes):
    values = dict(
        symbol="AAPL",
        side=Side.BUY,
        quantity=Decimal("1"),
        order_type=OrderType.MARKET,
    )
    values.update(changes)
    return ExecutionIntent(**values)


def _service(tmp_path: Path, adapter: InMemoryPaperAdapter) -> ExecutionService:
    return ExecutionService(
        adapter,
        policy_store=RiskPolicyStore(tmp_path / "policy.json"),
        ledger=ExecutionLedger(tmp_path / "ledger.jsonl"),
    )


def test_lifecycle_submit_query_and_cancel_is_provider_neutral(tmp_path: Path):
    adapter = InMemoryPaperAdapter()
    service = _service(tmp_path, adapter)
    intent = _intent(intent_id="lifecycle-1")

    accepted = service.submit(intent, _state(), human_authorized=True)
    assert accepted.status is ExecutionStatus.ACCEPTED
    assert service.get_status(accepted.execution_id) == accepted

    adapter.set_status(accepted.execution_id, ExecutionStatus.PARTIALLY_FILLED)
    assert service.get_status(accepted.execution_id).status is ExecutionStatus.PARTIALLY_FILLED
    adapter.set_status(accepted.execution_id, ExecutionStatus.FILLED)
    assert service.get_status(accepted.execution_id).status is ExecutionStatus.FILLED

    cancellation = service.cancel(accepted.execution_id, _state(), intent=intent)
    assert cancellation.status is ExecutionStatus.FILLED


def test_cancel_is_authorized_and_query_cannot_submit(tmp_path: Path):
    adapter = InMemoryPaperAdapter()
    service = _service(tmp_path, adapter)
    intent = _intent(intent_id="cancel-1")
    accepted = service.submit(intent, _state(), human_authorized=True)

    cancellation = service.cancel(accepted.execution_id, _state(), intent=intent)
    assert cancellation.status is ExecutionStatus.CANCEL_PENDING
    assert service.get_status(accepted.execution_id).status is ExecutionStatus.CANCEL_PENDING
    assert len(adapter._orders) == 1

    adapter.set_status(accepted.execution_id, ExecutionStatus.CANCELLED)
    assert service.get_status(accepted.execution_id).status is ExecutionStatus.CANCELLED


def test_risk_rejection_happens_before_adapter_for_submit_and_cancel(tmp_path: Path):
    adapter = InMemoryPaperAdapter()
    service = _service(tmp_path, adapter)
    intent = _intent(intent_id="risk-1")

    rejected = service.submit(intent, None, human_authorized=True)
    assert rejected.status is ExecutionStatus.REJECTED
    assert "MISSING_CRITICAL_RISK_DATA" in rejected.reason
    assert adapter._orders == {}

    accepted = service.submit(_intent(intent_id="risk-2"), _state(), human_authorized=True)
    rejected_cancel = service.cancel(accepted.execution_id, None)
    assert rejected_cancel.status is ExecutionStatus.REJECTED
    assert adapter.get_status(accepted.execution_id).status is ExecutionStatus.ACCEPTED
    assert [entry["kind"] for entry in service._ledger.entries()] == [
        "EXECUTION_INTENT", "RISK_DECISION", "EXECUTION_RESULT",
        "EXECUTION_INTENT", "RISK_DECISION", "EXECUTION_RESULT",
        "RISK_DECISION",
    ]


def test_real_cancel_is_rejected_before_adapter_and_audited(tmp_path: Path):
    adapter = InMemoryPaperAdapter()
    service = _service(tmp_path, adapter)
    intent = _intent(intent_id="real-cancel", mode=ExecutionMode.PAPER)
    accepted = service.submit(intent, _state(), human_authorized=True)

    real_intent = _intent(intent_id="real-cancel-request", mode=ExecutionMode.REAL)
    rejected = service.cancel(accepted.execution_id, _state(), intent=real_intent)

    assert rejected.status is ExecutionStatus.REJECTED
    assert "REAL_EXECUTION_DISABLED" in rejected.reason
    assert service.get_status(accepted.execution_id).status is ExecutionStatus.ACCEPTED
    assert [entry["kind"] for entry in service._ledger.entries()] == [
        "EXECUTION_INTENT", "RISK_DECISION", "EXECUTION_RESULT", "RISK_DECISION",
    ]


def test_statuses_and_unknown_errors_are_normalized_without_losing_original(tmp_path: Path):
    adapter = InMemoryPaperAdapter()
    service = _service(tmp_path, adapter)
    accepted = service.submit(_intent(intent_id="states-1"), _state(), human_authorized=True)

    for status in (ExecutionStatus.EXPIRED, ExecutionStatus.REJECTED, ExecutionStatus.UNKNOWN):
        adapter.set_status(accepted.execution_id, status)
        assert service.get_status(accepted.execution_id).status is status

    unknown = service.get_status("missing-execution")
    assert unknown.status is ExecutionStatus.UNKNOWN
    assert unknown.error is not None
    assert unknown.error.code == "EXECUTION_NOT_FOUND"
    assert isinstance(unknown.error.original_error, KeyError)


def test_same_intent_id_remains_idempotent_with_lifecycle_adapter(tmp_path: Path):
    adapter = InMemoryPaperAdapter()
    service = _service(tmp_path, adapter)
    intent = _intent(intent_id="same-intent")

    first = service.submit(intent, _state(), human_authorized=True)
    retry = service.submit(intent, _state(), human_authorized=True)
    assert retry == first
    assert len(adapter._orders) == 1
