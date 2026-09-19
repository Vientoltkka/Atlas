"""The enforced Intent -> RiskGate -> approved adapter execution facade."""
from __future__ import annotations

import uuid

from finance.execution.adapters import BrokerAdapter
from finance.execution.ledger import ExecutionLedger
from finance.execution.models import ExecutionIntent, ExecutionMode, ExecutionResult, ExecutionStatus
from finance.execution.policy import RiskPolicy, RiskPolicyStore
from finance.execution.risk_gate import PortfolioRiskState, RiskGate


class ExecutionService:
    def __init__(self, adapter: BrokerAdapter, *, gate: RiskGate | None = None, policy_store: RiskPolicyStore | None = None, ledger: ExecutionLedger | None = None) -> None:
        self._adapter, self._gate = adapter, gate or RiskGate()
        self._policy_store, self._ledger = policy_store or RiskPolicyStore(), ledger or ExecutionLedger()

    def policy(self) -> RiskPolicy: return self._policy_store.load()
    def set_kill_switch(self, active: bool) -> RiskPolicy:
        policy = self.policy()
        updated = RiskPolicy(**{**policy.to_dict(), "kill_switch": active, "allowed_asset_classes": tuple(policy.allowed_asset_classes)})
        self._policy_store.save(updated); return updated

    def submit(self, intent: ExecutionIntent, state: PortfolioRiskState | None) -> ExecutionResult:
        self._ledger.append("EXECUTION_INTENT", intent)
        decision = self._gate.evaluate(intent, state, self.policy())
        self._ledger.append("RISK_DECISION", decision)
        if not decision.approved:
            result = ExecutionResult(execution_id=f"rejected-{uuid.uuid4().hex}", intent_id=intent.intent_id, status=ExecutionStatus.REJECTED, adapter="none", mode=intent.mode, timestamp=decision.evaluated_at, symbol=intent.symbol, side=intent.side, quantity=intent.quantity, currency=intent.currency, reason=";".join(decision.reasons))
            self._ledger.append("EXECUTION_RESULT", result)
            return result
        if intent.mode is not ExecutionMode.PAPER:
            raise AssertionError("approved non-PAPER intent violates RiskGate invariant")
        result = self._adapter._execute_approved(intent)
        self._ledger.append("EXECUTION_RESULT", result)
        return result
