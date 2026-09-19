from finance.discovery_universe import DiscoveryUniverseStore

from finance.opportunity import FinanceOpportunity, OpportunitySignal

from finance.opportunity_service import OpportunityScanResult

from finance.signal_discovery import FinanceSignalDiscovery

from use_cases.finance_signal_discovery_chat import (

    FinanceSignalDiscoveryChat,

    handles_signal_discovery_prompt,

)





class OpportunityService:

    def __init__(self):

        self.calls = []



    def scan_symbols(self, symbols):

        self.calls.append(tuple(symbols))



        return OpportunityScanResult(

            opportunities=(

                FinanceOpportunity(

                    symbol="AAA",

                    signal=OpportunitySignal.CANDIDATE,

                    reasons=("trend=ALZA",),

                    evidence=("market_day=2026-09-18",),

                ),

            ),

            checked_symbols=("AAA", "BBB"),

            errors=(),

        )





def test_universe_is_persistent_and_deduplicated(tmp_path):

    store = DiscoveryUniverseStore(tmp_path)



    store.add("aaa")

    store.add("AAA")

    store.add("bbb")



    assert store.symbols() == ("AAA", "BBB")



    reloaded = DiscoveryUniverseStore(tmp_path)

    assert reloaded.symbols() == ("AAA", "BBB")





def test_universe_remove(tmp_path):

    store = DiscoveryUniverseStore(tmp_path)

    store.add("AAA")

    store.add("BBB")



    assert store.remove("aaa") == "AAA"

    assert store.symbols() == ("BBB",)





def test_discovery_uses_explicit_universe(tmp_path):

    universe = DiscoveryUniverseStore(tmp_path)

    universe.add("AAA")

    universe.add("BBB")



    opportunity = OpportunityService()

    discovery = FinanceSignalDiscovery(universe, opportunity)



    result = discovery.scan()



    assert opportunity.calls == [("AAA", "BBB")]

    assert result.universe_size == 2

    assert result.checked_symbols == ("AAA", "BBB")

    assert [item.symbol for item in result.opportunities] == ["AAA"]





def test_discovery_prompt_is_distinct_from_watchlist():

    assert handles_signal_discovery_prompt(

        "descubre oportunidades tacticas"

    )

    assert handles_signal_discovery_prompt(

        "busca senales tacticas"

    )

    assert handles_signal_discovery_prompt(

        "busca candidatos tacticos"

    )

    assert not handles_signal_discovery_prompt(

        "busca oportunidades tacticas en mi watchlist"

    )





def test_discovery_prompt_tolerates_windows_question_mark_degradation():

    assert handles_signal_discovery_prompt(

        "busca se?ales t?cticas"

    )



def test_chat_renders_candidate_and_no_order_boundary(tmp_path):

    universe = DiscoveryUniverseStore(tmp_path)

    universe.add("AAA")

    universe.add("BBB")



    chat = FinanceSignalDiscoveryChat(

        FinanceSignalDiscovery(universe, OpportunityService())

    )



    text = chat.handle("descubre oportunidades t?cticas")



    assert "[PAPER][DISCOVERY]" in text

    assert "AAA: CANDIDATE" in text

    assert "no son cotizaciones en tiempo real" in text

    assert "no crea ni ejecuta ordenes PAPER ni reales" in text


def test_watchlist_prompt_is_not_claimed_by_discovery():
    assert not handles_signal_discovery_prompt(
        "busca oportunidades tácticas en mi watchlist"
    )

def test_chat_can_add_show_and_remove_universe_symbol(tmp_path):
    universe = DiscoveryUniverseStore(tmp_path)

    class EmptyOpportunityService:
        def scan_symbols(self, symbols):
            return OpportunityScanResult(
                opportunities=(),
                checked_symbols=tuple(symbols),
                errors=(),
            )

    chat = FinanceSignalDiscoveryChat(
        FinanceSignalDiscovery(
            universe,
            EmptyOpportunityService(),
        ),
        universe,
    )

    added = chat.handle("anade VUSA.AMS al universo Discovery")
    assert "VUSA.AMS" in added
    assert "anadido" in added
    assert universe.symbols() == ("VUSA.AMS",)

    shown = chat.handle("muestra el universo Discovery")
    assert "VUSA.AMS" in shown
    assert universe.symbols() == ("VUSA.AMS",)

    removed = chat.handle("elimina VUSA.AMS del universo Discovery")
    assert "VUSA.AMS" in removed
    assert "eliminado" in removed
    assert universe.symbols() == ()
