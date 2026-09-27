"""The enforced Intent -> RiskGate -> approved adapter execution facade."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from decimal import Decimal

from finance.execution.adapters import BrokerAdapter
from finance.execution.ledger import ExecutionLedger
from finance.execution.models import ExecutionIntent, ExecutionMode, ExecutionResult, ExecutionStatus
from finance.execution.policy import RiskPolicy, RiskPolicyStore
from finance.execution.risk_gate import PortfolioRiskState, RiskGate
from finance.paper.models import Side


class ExecutionService:
    def __init__(self, adapter: BrokerAdapter, *, gate: RiskGate | None = None, policy_store: RiskPolicyStore | None = None, ledger: ExecutionLedger | None = None) -> None:
        self._adapter, self._gate = adapter, gate or RiskGate()
        self._policy_store, self._ledger = policy_store or RiskPolicyStore(), ledger or ExecutionLedger()
        self._submit_lock = threading.RLock()
        self._intents_by_execution_id: dict[str, ExecutionIntent] = {}

    def policy(self) -> RiskPolicy: return self._policy_store.load()
    def set_kill_switch(self, active: bool) -> RiskPolicy:
        policy = self.policy()
        updated = RiskPolicy(**{**policy.to_dict(), "kill_switch": active, "allowed_asset_classes": tuple(policy.allowed_asset_classes)})
        self._policy_store.save(updated); return updated

    @staticmethod
    def _intent_payload(intent: ExecutionIntent) -> dict[str, object]:
        return {
            "symbol": intent.symbol,
            "side": intent.side.value,
            "quantity": str(intent.quantity),
            "order_type": intent.order_type.value,
            "mode": intent.mode.value,
            "reference_price": str(intent.reference_price) if intent.reference_price is not None else None,
            "strategy": intent.strategy,
            "signal_id": intent.signal_id,
            "asset_class": intent.asset_class,
            "currency": intent.currency,
            "intent_id": intent.intent_id,
            "created_at": intent.created_at.isoformat(),
        }

    @staticmethod
    def _result_from_payload(payload: dict[str, object]) -> ExecutionResult:
        return ExecutionResult(
            execution_id=str(payload["execution_id"]),
            intent_id=str(payload["intent_id"]),
            status=ExecutionStatus(str(payload["status"])),
            adapter=str(payload["adapter"]),
            mode=ExecutionMode(str(payload["mode"])),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            symbol=str(payload["symbol"]),
            side=Side(str(payload["side"])),
            quantity=Decimal(str(payload["quantity"])),
            price=Decimal(str(payload["price"])) if payload.get("price") is not None else None,
            commission=Decimal(str(payload["commission"])) if payload.get("commission") is not None else None,
            currency=str(payload["currency"]) if payload.get("currency") is not None else None,
            reason=str(payload.get("reason", "")),
        )

    def _approved(self, intent: ExecutionIntent, state: PortfolioRiskState | None) -> ExecutionResult | None:
        decision = self._gate.evaluate(intent, state, self.policy())
        if decision.approved:
            return None
        return ExecutionResult(
            execution_id=f"rejected-{uuid.uuid4().hex}",
            intent_id=intent.intent_id,
            status=ExecutionStatus.REJECTED,
            adapter="none",
            mode=intent.mode,
            timestamp=decision.evaluated_at,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            currency=intent.currency,
            reason=";".join(decision.reasons),
        )

    def _prior_result(self, intent: ExecutionIntent) -> tuple[ExecutionResult | None, bool]:
        expected = self._intent_payload(intent)
        for entry in self._ledger.entries():
            payload = entry.get("payload", {})
            if entry.get("kind") != "EXECUTION_INTENT" or payload.get("intent_id") != intent.intent_id:
                continue
            prior = dict(payload)
            prior.pop("intent_id", None)
            current = dict(expected)
            current.pop("intent_id", None)
            if json.dumps(prior, sort_keys=True) != json.dumps(current, sort_keys=True):
                return None, True
            for result_entry in self._ledger.entries():
                result_payload = result_entry.get("payload", {})
                if result_entry.get("kind") == "EXECUTION_RESULT" and result_payload.get("intent_id") == intent.intent_id:
                    return self._result_from_payload(result_payload), False
            return None, False
        return None, False

    def submit(
        self,
        intent: ExecutionIntent,
        state: PortfolioRiskState | None,
        *,
        human_authorized: bool = False,
    ) -> ExecutionResult:
        if not human_authorized:
            return ExecutionResult(
                execution_id=f"rejected-{uuid.uuid4().hex}",
                intent_id=intent.intent_id,
                status=ExecutionStatus.REJECTED,
                adapter="none",
                mode=intent.mode,
                timestamp=intent.created_at,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                currency=intent.currency,
                reason="HUMAN_AUTHORIZATION_REQUIRED",
            )
        with self._submit_lock:
            prior_result, content_mismatch = self._prior_result(intent)
            if prior_result is not None:
                return prior_result
            if content_mismatch:
                self._ledger.append("EXECUTION_INTENT_CONFLICT", {"intent_id": intent.intent_id, "reason": "DUPLICATE_INTENT_CONTENT_MISMATCH", "requested": self._intent_payload(intent)})
                return ExecutionResult(execution_id=f"rejected-{uuid.uuid4().hex}", intent_id=intent.intent_id, status=ExecutionStatus.REJECTED, adapter="none", mode=intent.mode, timestamp=intent.created_at, symbol=intent.symbol, side=intent.side, quantity=intent.quantity, currency=intent.currency, reason="DUPLICATE_INTENT_CONTENT_MISMATCH")
            if any(entry.get("kind") == "EXECUTION_INTENT" and entry.get("payload", {}).get("intent_id") == intent.intent_id for entry in self._ledger.entries()):
                result = ExecutionResult(execution_id=f"rejected-{uuid.uuid4().hex}", intent_id=intent.intent_id, status=ExecutionStatus.REJECTED, adapter="none", mode=intent.mode, timestamp=intent.created_at, symbol=intent.symbol, side=intent.side, quantity=intent.quantity, currency=intent.currency, reason="DUPLICATE_INTENT_INCOMPLETE")
                self._ledger.append("EXECUTION_RESULT", result)
                return result
            self._ledger.append("EXECUTION_INTENT", intent)
            decision = self._gate.evaluate(intent, state, self.policy())
            self._ledger.append("RISK_DECISION", decision)
            if not decision.approved:
                result = ExecutionResult(execution_id=f"rejected-{uuid.uuid4().hex}", intent_id=intent.intent_id, status=ExecutionStatus.REJECTED, adapter="none", mode=intent.mode, timestamp=decision.evaluated_at, symbol=intent.symbol, side=intent.side, quantity=intent.quantity, currency=intent.currency, reason=";".join(decision.reasons))
                self._ledger.append("EXECUTION_RESULT", result)
                return result
            if intent.mode is not ExecutionMode.PAPER:
                raise AssertionError("approved non-PAPER intent violates RiskGate invariant")
            submit = getattr(self._adapter, "submit_order", None)
            result = submit(intent) if callable(submit) else self._adapter._execute_approved(intent)
            self._intents_by_execution_id[result.execution_id] = intent
            self._ledger.append("EXECUTION_RESULT", result)
            return result

    def get_status(self, execution_id: str) -> ExecutionResult:
        """Read-only status query; it cannot create or submit an order."""
        query = getattr(self._adapter, "get_status", None)
        if not callable(query):
            raise TypeError("adapter does not support status queries")
        return query(execution_id)

    def cancel(self, execution_id: str, state: PortfolioRiskState | None, *, intent: ExecutionIntent | None = None) -> ExecutionResult:
        """Cancel an accepted order only after explicit RiskGate authorization."""
        with self._submit_lock:
            known_intent = intent or self._intents_by_execution_id.get(execution_id)
            if known_intent is None:
                error = ValueError("intent requerido para autorizar la cancelacion")
                return ExecutionResult(
                    execution_id=execution_id,
                    intent_id="",
                    status=ExecutionStatus.UNKNOWN,
                    adapter=getattr(self._adapter, "name", "unknown"),
                    mode=ExecutionMode.PAPER,
                    timestamp=datetime.now().astimezone(),
                    symbol="UNKNOWN",
                    side=Side.BUY,
                    quantity=Decimal("0.00000001"),
                    reason="CANCEL_INTENT_REQUIRED",
                    error=ExecutionError("CANCEL_INTENT_REQUIRED", str(error), error),
                )
            decision = self._gate.evaluate(known_intent, state, self.policy())
            self._ledger.append("RISK_DECISION", decision)
            if not decision.approved or known_intent.mode is not ExecutionMode.PAPER:
                reasons = decision.reasons or ("REAL_EXECUTION_DISABLED",)
                return ExecutionResult(
                    execution_id=f"rejected-{uuid.uuid4().hex}",
                    intent_id=known_intent.intent_id,
                    status=ExecutionStatus.REJECTED,
                    adapter="none",
                    mode=known_intent.mode,
                    timestamp=decision.evaluated_at,
                    symbol=known_intent.symbol,
                    side=known_intent.side,
                    quantity=known_intent.quantity,
                    currency=known_intent.currency,
                    reason=";".join(reasons),
                )
            cancel = getattr(self._adapter, "cancel_order", None)
            if not callable(cancel):
                raise TypeError("adapter does not support cancellation")
            result = cancel(execution_id)
            if result.error is None:
                self._ledger.append("EXECUTION_CANCELLATION", result)
            else:
                self._ledger.append(
                    "EXECUTION_CANCELLATION",
                    {
                        "execution_id": result.execution_id,
                        "status": result.status.value,
                        "reason": result.reason,
                        "error": {
                            "code": result.error.code,
                            "message": result.error.message,
                        },
                    },
                )
            return result
