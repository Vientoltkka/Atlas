"""Persistent, conservative and deterministic execution limits."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class RiskPolicy:
    max_exposure_per_asset: Decimal = Decimal("0.20")
    max_total_exposure: Decimal = Decimal("0.60")
    max_open_positions: int = 10
    max_order_value: Decimal = Decimal("1000")
    max_daily_trades: int = 5
    max_daily_loss: Decimal = Decimal("500")
    allowed_asset_classes: tuple[str, ...] = ("EQUITY", "ETF")
    kill_switch: bool = False
    prohibit_leverage: bool = True
    prohibit_shorts: bool = True
    prohibit_derivatives: bool = True

    def __post_init__(self) -> None:
        for name in ("max_exposure_per_asset", "max_total_exposure"):
            value = Decimal(getattr(self, name))
            if not Decimal("0") < value <= Decimal("1"):
                raise ValueError(f"{name} debe estar entre 0 y 1")
            object.__setattr__(self, name, value)
        for name in ("max_order_value", "max_daily_loss"):
            value = Decimal(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} debe ser positivo")
            object.__setattr__(self, name, value)
        for name in ("max_open_positions", "max_daily_trades"):
            if isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} debe ser >= 1")
        if not self.allowed_asset_classes:
            raise ValueError("allowed_asset_classes vacio")

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        for key in ("max_exposure_per_asset", "max_total_exposure", "max_order_value", "max_daily_loss"):
            raw[key] = str(raw[key])
        return raw

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RiskPolicy":
        values = dict(data)
        for key in ("max_exposure_per_asset", "max_total_exposure", "max_order_value", "max_daily_loss"):
            values[key] = Decimal(str(values[key]))
        values["allowed_asset_classes"] = tuple(values["allowed_asset_classes"])
        return cls(**values)


class RiskPolicyStore:
    def __init__(self, path: Path | str = Path(".atlas") / "finance_execution" / "risk_policy.json") -> None:
        self.path = Path(path)

    def load(self) -> RiskPolicy:
        if not self.path.exists():
            policy = RiskPolicy()
            self.save(policy)
            return policy
        return RiskPolicy.from_dict(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, policy: RiskPolicy) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix=".risk_policy.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(policy.to_dict(), handle, ensure_ascii=False, sort_keys=True)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(raw_path, self.path)
        except BaseException:
            try: os.unlink(raw_path)
            except OSError: pass
            raise
