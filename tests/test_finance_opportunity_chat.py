from finance.opportunity import FinanceOpportunity, OpportunitySignal
from finance.opportunity_service import OpportunityScanResult
from use_cases.finance_opportunity_chat import (
    FinanceOpportunityChat,
    handles_finance_opportunity_prompt,
)


class Service:
    def scan_tactical_watchlist(self):
        return OpportunityScanResult(
            opportunities=(
                FinanceOpportunity(
                    symbol="TEST",
                    signal=OpportunitySignal.CANDIDATE,
                    reasons=("Atlas quantitative trend is ALZA",),
                    evidence=(
                        "market_day=2026-09-18",
                        "last_close=110",
                        "trend=ALZA",
                    ),
                ),
            ),
            checked_symbols=("TEST",),
            errors=(),
        )


def test_handles_explicit_opportunity_watchlist_prompt():
    assert handles_finance_opportunity_prompt(
        "busca oportunidades tácticas en mi watchlist"
    )


def test_does_not_capture_generic_watchlist_prompt():
    assert not handles_finance_opportunity_prompt(
        "revisa mi watchlist"
    )


def test_chat_renders_candidate_and_safety_boundary():
    response = FinanceOpportunityChat(Service()).handle(
        "busca oportunidades tácticas en mi watchlist"
    )

    assert "[PAPER][OPPORTUNITY]" in response
    assert "TEST: CANDIDATE" in response
    assert "market_day=2026-09-18" in response
    assert "no se ha creado ni ejecutado ninguna orden" in response


def test_opportunity_chat_remembers_candidates_and_warns_daily_not_realtime():
    class Result:
        checked_symbols = ("VUSA.AMS",)
        errors = ()

        class Candidate:
            symbol = "VUSA.AMS"
            reasons = ("trend=ALZA",)
            evidence = ("last_close=100",)

        opportunities = (Candidate(),)

    class Service:
        def scan_tactical_watchlist(self):
            return Result()

    chat = FinanceOpportunityChat(Service())
    text = chat.handle("busca oportunidades tacticas en mi watchlist")

    assert chat.last_candidates == ("VUSA.AMS",)
    assert "no son cotizaciones en tiempo real" in text
    assert "importa primero el cierre diario" in text
    assert "cantidad expl?cita" in text

def test_handles_strategy_performance_prompt():
    from use_cases.finance_opportunity_chat import (
        handles_strategy_performance_prompt,
    )

    assert handles_strategy_performance_prompt("rendimiento de estrategia")
    assert handles_strategy_performance_prompt("rendimiento de se\u00f1ales")
    assert not handles_strategy_performance_prompt("busca oportunidades")


def test_strategy_performance_with_open_signal(tmp_path):
    from datetime import date
    from decimal import Decimal

    from finance.strategy_evaluation import StrategyEvaluationStore
    from use_cases.finance_opportunity_chat import FinanceOpportunityChat

    store = StrategyEvaluationStore(tmp_path)
    store.record_candidate(
        symbol="VUSA.AMS",
        signal_day=date(2026, 9, 18),
        signal_close=Decimal("125.9990"),
        rule="trend=ALZA",
    )

    chat = FinanceOpportunityChat.__new__(FinanceOpportunityChat)
    chat._evaluation_store = store

    text = chat._render_strategy_performance()

    assert "Se?ales registradas: 1." in text
    assert "Evaluadas a 20 sesiones: 0." in text
    assert "Abiertas: 1." in text
    assert "no hay evidencia suficiente" in text
