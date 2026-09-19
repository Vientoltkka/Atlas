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
