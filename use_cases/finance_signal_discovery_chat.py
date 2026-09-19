"""Interfaz conversacional determinista de Finance Signal Discovery V1."""

from __future__ import annotations

import re
import unicodedata

from finance.discovery_universe import (
    DiscoveryUniverseError,
    DiscoveryUniverseStore,
)
from finance.signal_discovery import FinanceSignalDiscovery


_SYMBOL = r"[A-Za-z0-9.-]{1,20}"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    folded = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).lower()

    # Defensive compatibility with text degraded by Windows consoles.
    return (
        folded
        .replace("se?ales", "senales")
        .replace("se?al", "senal")
        .replace("t?cticas", "tacticas")
        .replace("t?ctica", "tactica")
    )


def handles_signal_discovery_prompt(prompt: str) -> bool:
    text = _fold(prompt)

    # La watchlist mantiene su propio flujo.
    if "watchlist" in text:
        return False

    if "universo discovery" in text:
        return True

    discovery_terms = (
        "descubre",
        "descubrir",
        "discovery",
        "busca senales",
        "buscar senales",
        "busca candidatos",
        "buscar candidatos",
    )

    return any(term in text for term in discovery_terms) and (
        "tactic" in text
        or "oportunidad" in text
        or "senal" in text
        or "candidato" in text
    )


class FinanceSignalDiscoveryChat:
    def __init__(
        self,
        discovery: FinanceSignalDiscovery,
        universe: DiscoveryUniverseStore | None = None,
    ) -> None:
        self._discovery = discovery
        self._universe = universe or discovery.universe

    @property
    def universe(self) -> DiscoveryUniverseStore:
        return self._universe

    def handles(self, prompt: str) -> bool:
        return handles_signal_discovery_prompt(prompt)

    def handle(self, prompt: str) -> str:
        text = _fold(prompt)

        if "universo discovery" in text:
            words = set(re.findall(r"[a-z0-9.-]+", text))

            if words.intersection(
                {"anade", "agrega", "agregar"}
            ):
                return self._add_symbol(prompt)

            if words.intersection(
                {"elimina", "eliminar", "quita", "quitar"}
            ):
                return self._remove_symbol(prompt)

            if words.intersection(
                {"muestra", "mostrar", "lista", "listar", "ver"}
            ):
                return self._show_universe()

        return self._scan()

    def _extract_symbol(self, prompt: str) -> str | None:
        prompt = _fold(prompt)

        patterns = (
            rf"(?:anade|agrega)\s+({_SYMBOL})\s+al\s+universo\s+discovery",
            rf"(?:elimina|quita)\s+({_SYMBOL})\s+del\s+universo\s+discovery",
        )

        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _add_symbol(self, prompt: str) -> str:
        symbol = self._extract_symbol(prompt)

        if not symbol:
            return (
                "[PAPER][DISCOVERY] Falta el provider_symbol exacto. "
                "Ejemplo: anade VUSA.AMS al universo Discovery."
            )

        try:
            stored = self._universe.add(symbol)
        except Exception as exc:
            return f"[PAPER][DISCOVERY] No se pudo a?adir el simbolo: {exc}"

        return (
            f"[PAPER][DISCOVERY] {stored} anadido al universo Discovery.\n"
            "Esto no crea ordenes ni modifica la cartera PAPER."
        )

    def _remove_symbol(self, prompt: str) -> str:
        symbol = self._extract_symbol(prompt)

        if not symbol:
            return (
                "[PAPER][DISCOVERY] Falta el provider_symbol exacto. "
                "Ejemplo: elimina VUSA.AMS del universo Discovery."
            )

        try:
            removed = self._universe.remove(symbol)
        except DiscoveryUniverseError as exc:
            return f"[PAPER][DISCOVERY] {exc}"
        except Exception as exc:
            return f"[PAPER][DISCOVERY] No se pudo eliminar el simbolo: {exc}"

        return (
            f"[PAPER][DISCOVERY] {removed} eliminado del universo Discovery.\n"
            "La watchlist y la cartera PAPER permanecen independientes."
        )

    def _show_universe(self) -> str:
        try:
            symbols = self._universe.symbols()
        except DiscoveryUniverseError as exc:
            return f"[PAPER][DISCOVERY] No se pudo leer el universo: {exc}"

        if not symbols:
            return (
                "[PAPER][DISCOVERY] Universo Discovery vacio.\n"
                "Anade provider_symbols explicitos antes de ejecutar Discovery."
            )

        return (
            "[PAPER][DISCOVERY] Universo Discovery:\n- "
            + "\n- ".join(symbols)
            + "\nNo implica ninguna orden ni posicion."
        )

    def _scan(self) -> str:
        try:
            result = self._discovery.scan()
        except DiscoveryUniverseError as exc:
            return f"[PAPER][DISCOVERY] No se pudo leer el universo: {exc}"

        lines = [
            "[PAPER][DISCOVERY] Escaneo tactico del universo Discovery.",
            f"Universo registrado: {result.universe_size}.",
        ]

        if result.checked_symbols:
            lines.append(
                "Simbolos revisados: "
                + ", ".join(result.checked_symbols)
                + "."
            )
        else:
            lines.append(
                "No hay simbolos disponibles para revisar en esta ejecucion."
            )

        if result.opportunities:
            lines.append("Candidatos detectados:")

            for opportunity in result.opportunities:
                lines.append(
                    f"- {opportunity.symbol}: CANDIDATE | "
                    + "; ".join(opportunity.reasons)
                )

                if opportunity.evidence:
                    lines.append(
                        "  Evidencia: "
                        + "; ".join(opportunity.evidence)
                    )
        else:
            lines.append(
                "No se detectaron CANDIDATE en los simbolos revisados."
            )

        if result.errors:
            lines.append("Errores controlados:")
            lines.extend(f"- {error}" for error in result.errors)

        lines.extend(
            (
                "Los datos utilizados son diarios; no son cotizaciones "
                "en tiempo real.",
                "Discovery registra senales para evaluacion de estrategia, "
                "pero no crea ni ejecuta ordenes PAPER ni reales.",
            )
        )

        return "\n".join(lines)
