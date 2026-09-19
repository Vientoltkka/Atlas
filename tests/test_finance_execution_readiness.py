from decimal import Decimal
from pathlib import Path

import pytest

from finance.execution.adapters import PaperExecutionAdapter
from finance.execution.ledger import ExecutionLedger
from finance.execution.models import ExecutionIntent, ExecutionMode, ExecutionStatus
from finance.execution.policy import RiskPolicy, RiskPolicyStore
from finance.execution.risk_gate import PortfolioRiskState, RiskGate
from finance.execution.service import ExecutionService
from finance.paper.models import MarketEvent, OrderType, Side
from finance.paper.service import PaperFinanceService
from finance.paper.store import PaperStore


def state(**changes):
    values = dict(nav=Decimal("10000"), cash=Decimal("10000"), positions={}, prices={"AAPL": Decimal("100"), "OPT": Decimal("10")}, daily_trade_count=0, daily_realized_loss=Decimal("0"))
    values.update(changes); return PortfolioRiskState(**values)


def intent(**changes):
    values = dict(symbol="AAPL", side=Side.BUY, quantity=Decimal("1"), order_type=OrderType.MARKET)
    values.update(changes); return ExecutionIntent(**values)


@pytest.mark.parametrize("changed, reason", [
    ({"quantity": Decimal("11")}, "MAX_ORDER_VALUE"),
    ({"positions": {"AAPL": Decimal("20")}}, "MAX_EXPOSURE_PER_ASSET"),
    ({"positions": {"MSFT": Decimal("60")}, "prices": {"AAPL": Decimal("100"), "MSFT": Decimal("100")}}, "MAX_TOTAL_EXPOSURE"),
    ({"positions": {str(i): Decimal("1") for i in range(10)}}, "MAX_OPEN_POSITIONS"),
    ({"daily_trade_count": 5}, "MAX_DAILY_TRADES"),
    ({"daily_realized_loss": Decimal("500")}, "MAX_DAILY_LOSS"),
])
def test_gate_rejects_each_limit(changed, reason):
    item_changes = {key: changed.pop(key) for key in tuple(changed) if key == "quantity"}
    decision = RiskGate().evaluate(intent(**item_changes), state(**changed), RiskPolicy())
    assert not decision.approved and reason in decision.reasons


def test_gate_approves_valid_paper_intent_and_fails_closed():
    assert RiskGate().evaluate(intent(), state(), RiskPolicy()).approved
    decision = RiskGate().evaluate(intent(), state(prices=None), RiskPolicy())
    assert not decision.approved and "MISSING_CRITICAL_RISK_DATA" in decision.reasons


@pytest.mark.parametrize("bad_intent, reason", [
    (intent(quantity=Decimal("101")), "NO_LEVERAGE"),
    (intent(side=Side.SELL), "NO_SHORTS"),
    (intent(symbol="OPT", asset_class="DERIVATIVE"), "NO_DERIVATIVES"),
    (intent(mode=ExecutionMode.REAL), "REAL_EXECUTION_DISABLED"),
])
def test_gate_prohibitions(bad_intent, reason):
    decision = RiskGate().evaluate(bad_intent, state(), RiskPolicy())
    assert not decision.approved and reason in decision.reasons


class SpyAdapter:
    name = "spy"
    def __init__(self): self.calls = 0
    def _execute_approved(self, item):
        self.calls += 1
        return __import__("finance.execution.models", fromlist=["ExecutionResult"]).ExecutionResult("x", item.intent_id, ExecutionStatus.FILLED, self.name, item.mode, item.created_at, item.symbol, item.side, item.quantity)


def test_service_enforces_gate_kill_switch_and_audits(tmp_path: Path):
    adapter = SpyAdapter()
    service = ExecutionService(adapter, policy_store=RiskPolicyStore(tmp_path / "policy.json"), ledger=ExecutionLedger(tmp_path / "ledger.jsonl"))
    rejected = service.submit(intent(mode=ExecutionMode.REAL), state())
    assert rejected.status is ExecutionStatus.REJECTED and adapter.calls == 0 and rejected.reason == "REAL_EXECUTION_DISABLED"
    service.set_kill_switch(True)
    killed = service.submit(intent(), state())
    assert "KILL_SWITCH_ACTIVE" in killed.reason and adapter.calls == 0
    service.set_kill_switch(False)
    approved = service.submit(intent(), state())
    assert approved.status is ExecutionStatus.FILLED and adapter.calls == 1
    entries = service._ledger.entries()
    assert [entry["kind"] for entry in entries] == ["EXECUTION_INTENT", "RISK_DECISION", "EXECUTION_RESULT"] * 3
    assert {entry["payload"]["intent_id"] for entry in entries if "intent_id" in entry["payload"]} == {approved.intent_id, rejected.intent_id, killed.intent_id}


def test_paper_adapter_only_runs_after_approved_service_path(tmp_path: Path):
    paper = PaperFinanceService(PaperStore(tmp_path / "paper"))
    paper.record_market_event(MarketEvent(symbol="AAPL", price=Decimal("100")))
    service = ExecutionService(PaperExecutionAdapter(paper), policy_store=RiskPolicyStore(tmp_path / "policy.json"), ledger=ExecutionLedger(tmp_path / "ledger.jsonl"))
    assert service.submit(intent(), state()).status is ExecutionStatus.FILLED
    assert service.submit(intent(mode=ExecutionMode.REAL), state()).status is ExecutionStatus.REJECTED
    assert len(paper.fills()) == 1


def test_discovery_candidate_does_not_import_or_create_execution_intent():
    source = Path("finance/signal_discovery.py").read_text(encoding="utf-8")
    assert "ExecutionIntent" not in source and "finance.execution" not in source
