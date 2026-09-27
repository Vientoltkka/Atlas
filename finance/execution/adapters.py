"""Provider-neutral adapter boundary; only PAPER is implemented."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol

from finance.execution.models import ExecutionError, ExecutionIntent, ExecutionMode, ExecutionResult, ExecutionStatus
from finance.paper.models import PaperOrder
from finance.paper.service import PaperFinanceService
from finance.paper.models import Side


class BrokerAdapter(Protocol):
    name: str
    def submit_order(self, intent: ExecutionIntent) -> ExecutionResult: ...
    def get_status(self, execution_id: str) -> ExecutionResult: ...
    def cancel_order(self, execution_id: str) -> ExecutionResult: ...
    def _execute_approved(self, intent: ExecutionIntent) -> ExecutionResult: ...


def _error_result(execution_id: str, code: str, exc: BaseException) -> ExecutionResult:
    return ExecutionResult(
        execution_id=execution_id,
        intent_id="",
        status=ExecutionStatus.UNKNOWN,
        adapter="memory",
        mode=ExecutionMode.PAPER,
        timestamp=datetime.now(timezone.utc),
        symbol="UNKNOWN",
        side=Side.BUY,
        quantity=Decimal("0.00000001"),
        reason=code,
        error=ExecutionError(code, str(exc), exc),
    )


class PaperExecutionAdapter:
    name = "paper"
    def __init__(self, service: PaperFinanceService) -> None: self._service = service

    def _execute_approved(self, intent: ExecutionIntent) -> ExecutionResult:
        limit_price = intent.reference_price if intent.order_type.value == "LIMIT" else None
        order = PaperOrder(symbol=intent.symbol, side=intent.side, qty=intent.quantity, order_type=intent.order_type, limit_price=limit_price)
        decision = self._service.execute_confirmed_order(order)
        fill = decision.fill
        status = ExecutionStatus(decision.status.value)
        return ExecutionResult(execution_id=fill.fill_id if fill else order.order_id, intent_id=intent.intent_id, status=status, adapter=self.name, mode=intent.mode, timestamp=order.timestamp, symbol=intent.symbol, side=intent.side, quantity=intent.quantity, price=fill.price if fill else None, commission=fill.commission if fill else None, currency=intent.currency, reason=decision.reason)


class InMemoryPaperAdapter:
    """Synthetic provider-neutral adapter; it never talks to a broker."""

    name = "paper-memory"

    def __init__(self) -> None:
        self._orders: dict[str, ExecutionResult] = {}

    def submit_order(self, intent: ExecutionIntent) -> ExecutionResult:
        execution_id = f"paper-exec-{uuid.uuid4().hex}"
        result = ExecutionResult(
            execution_id=execution_id,
            intent_id=intent.intent_id,
            status=ExecutionStatus.ACCEPTED,
            adapter=self.name,
            mode=intent.mode,
            timestamp=datetime.now(timezone.utc),
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            currency=intent.currency,
            reason="synthetic order accepted",
        )
        self._orders[execution_id] = result
        return result

    def get_status(self, execution_id: str) -> ExecutionResult:
        result = self._orders.get(execution_id)
        if result is None:
            error = KeyError(execution_id)
            return _error_result(execution_id, "EXECUTION_NOT_FOUND", error)
        return result

    def cancel_order(self, execution_id: str) -> ExecutionResult:
        result = self.get_status(execution_id)
        if result.status is ExecutionStatus.UNKNOWN:
            return result
        if result.status in (ExecutionStatus.CANCELLED, ExecutionStatus.FILLED, ExecutionStatus.EXPIRED, ExecutionStatus.REJECTED):
            return result
        cancelled = ExecutionResult(
            **{**result.__dict__, "status": ExecutionStatus.CANCEL_PENDING, "reason": "synthetic cancellation pending"}
        )
        self._orders[execution_id] = cancelled
        return cancelled

    def set_status(self, execution_id: str, status: ExecutionStatus, *, reason: str = "synthetic status") -> ExecutionResult:
        result = self._orders[execution_id]
        updated = ExecutionResult(**{**result.__dict__, "status": status, "reason": reason})
        self._orders[execution_id] = updated
        return updated

    def _execute_approved(self, intent: ExecutionIntent) -> ExecutionResult:
        return self.submit_order(intent)
