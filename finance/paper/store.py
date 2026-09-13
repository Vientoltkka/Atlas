"""Persistencia JSON atomica y versionada del estado paper.

Almacena en .atlas/finance_paper/. Desde V2.7 el estado es version 2 e
incluye los modos paper (CORE/TACTICAL) con su InvestmentPolicy; los
estados version 1 (V2.1-V2.6, cartera unica) siguen siendo legibles para
migracion. La escritura es atomica (temporal + os.replace) y la lectura
es segura: corrupcion o version invalida producen un error controlado sin
sobrescribir ni modificar el fichero existente.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

STATE_VERSION = 2

SUPPORTED_VERSIONS = (1, 2)

STATE_FILE = "state.json"


class StoreError(Exception):
    """Error seguro de persistencia paper (corrupcion o version invalida)."""


class PaperStore:
    def __init__(self, directory: Path | str = Path(".atlas") / "finance_paper") -> None:
        self._directory = Path(directory)
        self._state_path = self._directory / STATE_FILE

    @property
    def state_path(self) -> Path:
        return self._state_path

    def save(self, data: Mapping[str, Any]) -> None:
        payload = dict(data)
        payload["schema_version"] = STATE_VERSION
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

    def load(self) -> dict[str, Any]:
        if not self._state_path.exists():
            raise StoreError(f"estado paper inexistente: {self._state_path}")
        raw = self._state_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(f"estado paper corrupto: {exc}") from exc
        if not isinstance(data, dict):
            raise StoreError("estado paper corrupto: raiz no es un objeto")
        version = data.get("schema_version")
        if version not in SUPPORTED_VERSIONS:
            raise StoreError(f"version de estado no soportada: {version!r}")
        return data

    def exists(self) -> bool:
        return self._state_path.exists()
