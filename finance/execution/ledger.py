"""Append-only JSONL audit ledger with atomic single-record appends."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path


def _json(value: object) -> object:
    if isinstance(value, (Decimal, datetime)): return str(value) if isinstance(value, Decimal) else value.isoformat()
    if hasattr(value, "value"): return value.value
    if is_dataclass(value): return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict): return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json(item) for item in value]
    return value


class ExecutionLedger:
    def __init__(self, path: Path | str = Path(".atlas") / "finance_execution" / "ledger.jsonl") -> None:
        self.path = Path(path)

    def append(self, kind: str, payload: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"kind": kind, "payload": _json(payload)}, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())

    def entries(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists(): return ()
        return tuple(json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line)
