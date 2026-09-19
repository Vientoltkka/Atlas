"""Universo persistente para Finance Signal Discovery V1.

Separado de watchlist y PAPER. Contiene provider_symbols expl?citos que Atlas
puede analizar mediante el pipeline cuantitativo existente. No crea ?rdenes,
MarketEvents ni posiciones.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from finance.watchlist.store import normalize_provider_symbol


SCHEMA_VERSION = 1
STATE_FILE = "universe.json"
DEFAULT_DIRECTORY = Path(".atlas") / "finance_discovery"


class DiscoveryUniverseError(Exception):
    pass


class DiscoveryUniverseStore:
    def __init__(self, directory: Path | str = DEFAULT_DIRECTORY) -> None:
        self._directory = Path(directory)
        self._state_path = self._directory / STATE_FILE

    @property
    def state_path(self) -> Path:
        return self._state_path

    def symbols(self) -> tuple[str, ...]:
        if not self._state_path.exists():
            return ()

        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiscoveryUniverseError(
                f"universo Discovery ilegible: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise DiscoveryUniverseError("universo Discovery corrupto")

        if data.get("schema_version") != SCHEMA_VERSION:
            raise DiscoveryUniverseError(
                f"versi?n Discovery no soportada: {data.get('schema_version')!r}"
            )

        raw = data.get("symbols", [])
        if not isinstance(raw, list):
            raise DiscoveryUniverseError(
                "universo Discovery corrupto: 'symbols' no es lista"
            )

        result = []
        seen = set()

        for item in raw:
            try:
                symbol = normalize_provider_symbol(str(item))
            except Exception as exc:
                raise DiscoveryUniverseError(
                    f"provider_symbol Discovery inv?lido: {item!r}"
                ) from exc

            if symbol not in seen:
                seen.add(symbol)
                result.append(symbol)

        return tuple(result)

    def add(self, provider_symbol: str) -> str:
        symbol = normalize_provider_symbol(provider_symbol)
        current = list(self.symbols())

        if symbol in current:
            return symbol

        current.append(symbol)
        self._save(current)
        return symbol

    def remove(self, provider_symbol: str) -> str:
        symbol = normalize_provider_symbol(provider_symbol)
        current = list(self.symbols())

        if symbol not in current:
            raise DiscoveryUniverseError(
                f"{symbol} no est? en el universo Discovery"
            )

        current.remove(symbol)
        self._save(current)
        return symbol

    def _save(self, symbols: list[str]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "symbols": symbols,
        }

        self._directory.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{STATE_FILE}.",
            suffix=".tmp",
            dir=str(self._directory),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(tmp_path, self._state_path)

        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
