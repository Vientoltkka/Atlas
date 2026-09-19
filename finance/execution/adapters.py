"""Provider-neutral adapter boundary; only PAPER is implemented."""
from __future__ import annotations

from typing import Protocol

from finance.execution.models import ExecutionIntent, ExecutionResult, ExecutionStatus
from finance.paper.models import PaperOrder
from finance.paper.service import PaperFinanceService


class BrokerAdapter(Protocol):
    name: str
    def _execute_approved(self, intent: ExecutionIntent) -> ExecutionResult: ...


class PaperExecutionAdapter:
    name = "paper"
    def __init__(self, service: PaperFinanceService) -> None: self._service = service

    def _execute_approved(self, intent: ExecutionIntent) -> ExecutionResult:
        order = PaperOrder(symbol=intent.symbol, side=intent.side, qty=intent.quantity, order_type=intent.order_type, limit_price=intent.reference_price)
        decision = self._service.execute_confirmed_order(order)
        fill = decision.fill
        status = ExecutionStatus(decision.status.value)
        return ExecutionResult(execution_id=fill.fill_id if fill else order.order_id, intent_id=intent.intent_id, status=status, adapter=self.name, mode=intent.mode, timestamp=order.timestamp, symbol=intent.symbol, side=intent.side, quantity=intent.quantity, price=fill.price if fill else None, commission=fill.commission if fill else None, currency=intent.currency, reason=decision.reason)
