"""Persistencia JSON atomica y versionada de la watchlist tactica.

Almacena en .atlas/finance_watchlist/watchlist.json, separada de la
cartera PAPER (.atlas/finance_paper/): ficheros y directorios distintos,
de modo que ninguna operacion de watchlist pueda tocar el estado paper.
La escritura es atomica (temporal + os.replace + fsync) y la lectura es
segura: corrupcion o version invalida producen un error controlado sin
sobrescribir el fichero existente.

La clave de cada entrada es el `provider_symbol` EXACTO tal y como el
proveedor (Alpha Vantage) lo devuelve y lo entiende: "VUSA.AMS" se
consulta como "VUSA.AMS" e "IBM" como "IBM". El sistema nunca elimina ni
altera sufijos de bolsa validos y nunca inventa un sufijo que el usuario
no haya dado. Las entradas heredadas del esquema V2.8 original
(simbolo + bolsa separados) se migran automaticamente a la carga:

- Si el identificador completo "SIMBOLO.BOLSA" era exactamente lo que
  guardo el usuario (por ejemplo VUSA + AMS -> VUSA.AMS), la migracion
  reconstruye el provider_symbol original sin perder el registro.

- Si el sufijo de bolsa no es verificable como parte del identificador
  del proveedor (por ejemplo IBM + NYSE, donde Alpha Vantage devuelve
  "IBM" a secas), la entrada se conserva con la marca `needs_review`
  para que la lista muestre un aviso claro por activo y el usuario pueda
  eliminarla o darla de alta de nuevo con el identificador exacto; el
  sistema NO inventa una conversion universal.

El mercado/bolsa, si llega a existir como metadato, es solo informativo:
no modifica el provider_symbol ni participa en la clave.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from finance.paper.policy import PaperMode

SCHEMA_VERSION = 2

LEGACY_SCHEMA_VERSION = 1

SUPPORTED_VERSIONS = (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION)

STATE_FILE = "watchlist.json"

_NOTE_MAX_LENGTH = 120

_SYMBOL_MAX_LENGTH = 12


class WatchlistStoreError(Exception):
    """Error seguro de persistencia de watchlist (validacion, corrupcion o version)."""


def normalize_note(note: str | None) -> str | None:
    """Normaliza una nota opcional: recortada, sin saltos y con limite."""
    if note is None:
        return None
    cleaned = " ".join(note.split())
    if not cleaned:
        return None
    if len(cleaned) > _NOTE_MAX_LENGTH:
        raise WatchlistStoreError(
            f"nota demasiado larga ({len(cleaned)} > {_NOTE_MAX_LENGTH} caracteres)"
        )
    return cleaned


def normalize_provider_symbol(symbol: str) -> str:
    """Normaliza un provider_symbol exacto sin alterar sufijos validos."""
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise WatchlistStoreError(f"provider_symbol vacio: {symbol!r}")
    if len(cleaned) > _SYMBOL_MAX_LENGTH:
        raise WatchlistStoreError(
            f"provider_symbol demasiado largo: {symbol!r}"
        )
    if any(character.isspace() for character in cleaned):
        raise WatchlistStoreError(f"provider_symbol con espacios: {symbol!r}")
    if cleaned.startswith(".") or cleaned.endswith(".") or ".." in cleaned:
        raise WatchlistStoreError(
            f"provider_symbol invalido: {symbol!r}"
        )
    return cleaned


class WatchlistStore:
    """Guarda y carga la watchlist como JSON versionado en un solo fichero."""

    def __init__(
        self, directory: Path | str = Path(".atlas") / "finance_watchlist"
    ) -> None:
        self._directory = Path(directory)
        self._state_path = self._directory / STATE_FILE

    @property
    def state_path(self) -> Path:
        return self._state_path

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Devuelve las entradas validadas en orden de alta; vacio si no hay fichero."""
        if not self._state_path.exists():
            return ()
        raw_entries = self._raw_entries()
        entries, migrated = _migrate_entries(raw_entries)
        if migrated:
            # Persiste la migracion sin perder ningun registro heredado.
            self._save_entries(entries)
        return tuple(entries)

    def add_entry(
        self,
        *,
        provider_symbol: str,
        mode: PaperMode,
        note: str | None = None,
        added_on: str,
    ) -> dict[str, Any]:
        """Anade una entrada validada; rechaza duplicados por provider_symbol."""
        entry = _validated_entry(
            provider_symbol=provider_symbol,
            mode=mode,
            note=note,
            added_on=added_on,
        )
        current = list(self.entries())
        key = entry["provider_symbol"]
        for existing in current:
            if existing["provider_symbol"] == key:
                raise WatchlistStoreError(
                    f"el simbolo {key} ya esta en seguimiento"
                )
        current.append(entry)
        self._save_entries(current)
        return entry

    def mark_reviewed(self, provider_symbol: str, reviewed_on: str) -> dict[str, Any]:
        """Guarda la fecha de ultima revision y limpia la marca de revision.

        Una revision exitosa ante el proveedor verifica que el
        provider_symbol es un identificador valido; por eso limpia
        needs_review. Maximo 1 consulta/dia.
        """
        target = normalize_provider_symbol(provider_symbol)
        current = list(self.entries())
        for index, existing in enumerate(current):
            if existing["provider_symbol"] == target:
                updated = dict(existing)
                updated["last_reviewed"] = reviewed_on
                updated["needs_review"] = False
                current[index] = updated
                self._save_entries(current)
                return updated
        raise WatchlistStoreError(
            f"el simbolo {target} no esta en seguimiento"
        )

    def remove_entry(self, provider_symbol: str) -> dict[str, Any]:
        """Elimina la entrada por provider_symbol; error controlado si no existe."""
        target = normalize_provider_symbol(provider_symbol)
        current = list(self.entries())
        for index, existing in enumerate(current):
            if existing["provider_symbol"] == target:
                removed = current.pop(index)
                self._save_entries(current)
                return removed
        raise WatchlistStoreError(
            f"el simbolo {target} no esta en seguimiento"
        )

    def load(self) -> dict[str, Any]:
        """Lee y valida el documento completo; error controlado si esta corrupto."""
        if not self._state_path.exists():
            raise WatchlistStoreError(
                f"watchlist inexistente: {self._state_path}"
            )
        raw = self._state_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WatchlistStoreError(f"watchlist corrupta: {exc}") from exc
        if not isinstance(data, dict):
            raise WatchlistStoreError("watchlist corrupta: raiz no es un objeto")
        version = data.get("schema_version")
        if version not in SUPPORTED_VERSIONS:
            raise WatchlistStoreError(f"version de watchlist no soportada: {version!r}")
        return data

    def exists(self) -> bool:
        return self._state_path.exists()

    def _raw_entries(self) -> list[Mapping[str, Any]]:
        data = self.load()
        raw_entries = data.get("entries")
        if raw_entries is None:
            return []
        if not isinstance(raw_entries, list):
            raise WatchlistStoreError("watchlist corrupta: 'entries' no es una lista")
        for item in raw_entries:
            if not isinstance(item, Mapping):
                raise WatchlistStoreError("watchlist corrupta: entrada no es un objeto")
        return list(raw_entries)

    def _save_entries(self, entries: list[dict[str, Any]]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entries": entries,
        }
        self._directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{STATE_FILE}.", suffix=".tmp", dir=str(self._directory)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._state_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _migrate_entries(
    raw_entries: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Convierte entradas heredadas (symbol+exchange) a provider_symbol.

    Devuelve (entradas migradas, hubo_migracion). Los registros nunca se
    pierden: el identificador exacto original del usuario era
    "SIMBOLO.BOLSA", asi que se reconstruye tal cual y la entrada queda
    marcada needs_review=True hasta que una revision la verifique con
    exito ante el proveedor. Si el sufijo no es verificable (por ejemplo
    IBM + NYSE, donde Alpha Vantage devuelve "IBM" a secas), la entrada
    conserva el aviso y el usuario puede eliminarla o reemplazarla; el
    sistema NO inventa una conversion universal.
    """
    migrated_entries: list[dict[str, Any]] = []
    migrated_any = False
    for item in raw_entries:
        if "provider_symbol" in item:
            entry = _validated_entry(
                provider_symbol=str(item.get("provider_symbol", "")),
                mode=PaperMode(str(item.get("mode", ""))),
                note=item.get("note"),
                added_on=str(item.get("added_on", "")),
                last_reviewed=item.get("last_reviewed"),
                needs_review=bool(item.get("needs_review", False)),
            )
            migrated_entries.append(entry)
            continue
        legacy_symbol = str(item.get("symbol", "")).strip().upper()
        legacy_exchange = str(item.get("exchange", "")).strip().upper()
        mode = PaperMode(str(item.get("mode", "")))
        note = item.get("note")
        added_on = str(item.get("added_on", ""))
        last_reviewed = item.get("last_reviewed")
        migrated_any = True
        if legacy_exchange and "." not in legacy_symbol:
            # El identificador exacto que el usuario guardo era
            # "SIMBOLO.BOLSA": se reconstruye sin perder el registro y
            # queda pendiente de verificacion ante el proveedor.
            provider_symbol = normalize_provider_symbol(
                f"{legacy_symbol}.{legacy_exchange}"
            )
        else:
            provider_symbol = normalize_provider_symbol(legacy_symbol)
        migrated_entries.append(
            _validated_entry(
                provider_symbol=provider_symbol,
                mode=mode,
                note=note,
                added_on=added_on,
                last_reviewed=last_reviewed,
                needs_review=True,
            )
        )
    return migrated_entries, migrated_any


def _validated_entry(
    *,
    provider_symbol: str,
    mode: PaperMode,
    note: str | None,
    added_on: str,
    last_reviewed: str | None = None,
    needs_review: bool = False,
) -> dict[str, Any]:
    clean_symbol = normalize_provider_symbol(provider_symbol)
    if not isinstance(mode, PaperMode):
        raise WatchlistStoreError(f"modo invalido: {mode!r}")
    clean_note = normalize_note(note)
    if not added_on:
        raise WatchlistStoreError("fecha de alta obligatoria")
    try:
        parsed_date = date.fromisoformat(added_on)
    except ValueError as exc:
        raise WatchlistStoreError(
            f"fecha de alta invalida: {added_on!r}"
        ) from exc
    if last_reviewed is not None:
        try:
            last_reviewed = date.fromisoformat(str(last_reviewed)).isoformat()
        except ValueError as exc:
            raise WatchlistStoreError(
                f"fecha de revision invalida: {last_reviewed!r}"
            ) from exc
    return {
        "provider_symbol": clean_symbol,
        "mode": mode.value,
        "note": clean_note,
        "added_on": parsed_date.isoformat(),
        "last_reviewed": last_reviewed,
        "needs_review": bool(needs_review),
    }
