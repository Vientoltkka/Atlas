"""Small, approval-gated repair workflow for Atlas self-improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping


class ImprovementClassification(str, Enum):
    REUSE = "REUSE"
    SKILL_GAP = "SKILL_GAP"
    CODE_REPAIR = "CODE_REPAIR"
    CAPABILITY_GAP = "CAPABILITY_GAP"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"


class RepairState(str, Enum):
    PROPOSED = "PROPOSED"
    AUTHORIZED = "AUTHORIZED"
    VALIDATED = "VALIDATED"
    ACCEPTED = "ACCEPTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True, slots=True)
class RepairProposal:
    """An exact, reviewable repair surface. It contains no executable commands."""

    proposal_id: str
    objective: str
    files: Mapping[str, str]
    focused_tests: tuple[str, ...]
    metric_directions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.objective or not self.files:
            raise ValueError("proposal_id, objective and files are required.")
        if not all(isinstance(path, str) and isinstance(content, str) for path, content in self.files.items()):
            raise TypeError("files must map relative paths to text content.")
        if not all(direction in {"decrease", "increase", "equal"} for direction in self.metric_directions.values()):
            raise ValueError("metric directions must be decrease, increase or equal.")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(self, "metric_directions", MappingProxyType(dict(self.metric_directions)))

    @property
    def authorization(self) -> str:
        payload = json.dumps(
            {
                "proposal_id": self.proposal_id,
                "files": dict(self.files),
                "tests": self.focused_tests,
                "metrics": dict(self.metric_directions),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "AUTORIZAR " + self.proposal_id + " " + hashlib.sha256(payload).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class RepairValidation:
    """Trusted validation output collected by the host, never by proposal content."""

    passed: bool
    before_metrics: Mapping[str, float] = field(default_factory=dict)
    after_metrics: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""


class SupervisedRepairWorkflow:
    """Apply a pre-inspected minimal repair only after explicit approval."""

    def __init__(self, project_root: Path, *, validator: Callable[[RepairProposal], RepairValidation]) -> None:
        self._root = project_root.resolve()
        self._validator = validator
        self._proposal: RepairProposal | None = None
        self._originals: dict[str, str | None] = {}
        self._state: RepairState | None = None
        self._authorization_consumed = False
        self.audit_log: list[dict[str, object]] = []

    @property
    def state(self) -> RepairState | None:
        return self._state

    @staticmethod
    def classify(prompt: str, *, reusable: bool = False) -> ImprovementClassification:
        text = " ".join(prompt.casefold().split()) if isinstance(prompt, str) else ""
        if not text or text in {"mejorate", "mejorate entero", "mejora todo"}:
            return ImprovementClassification.CLARIFICATION_REQUIRED
        if reusable:
            return ImprovementClassification.REUSE
        if any(term in text for term in ("corrige", "repara", "fallo", "fallos", "latencia")):
            return ImprovementClassification.CODE_REPAIR
        if "skill" in text:
            return ImprovementClassification.SKILL_GAP
        if any(term in text for term in ("herramienta", "proveedor", "infraestructura", "integracion")):
            return ImprovementClassification.CAPABILITY_GAP
        return ImprovementClassification.CLARIFICATION_REQUIRED

    def propose(self, proposal: RepairProposal) -> RepairProposal:
        """Record a bounded proposal and its exact pre-change snapshot, without writes."""
        self._validate_scope(proposal)
        self._proposal = proposal
        self._originals = {
            relative: self._target(relative).read_text(encoding="utf-8") if self._target(relative).exists() else None
            for relative in proposal.files
        }
        self._state = RepairState.PROPOSED
        self._authorization_consumed = False
        self._record("proposed", proposal)
        return proposal

    def authorize_and_apply(self, authorization: str) -> bool:
        proposal = self._proposal
        if proposal is None or self._state is not RepairState.PROPOSED:
            return False
        if self._authorization_consumed or not hmac.compare_digest(authorization, proposal.authorization):
            self._record("authorization_rejected", proposal)
            return False
        self._authorization_consumed = True
        self._state = RepairState.AUTHORIZED
        for relative, content in proposal.files.items():
            target = self._target(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self._record("applied", proposal)
        return True

    def validate(self) -> RepairValidation:
        proposal = self._require_state(RepairState.AUTHORIZED)
        result = self._validator(proposal)
        if not result.passed or not self._metrics_improved(proposal, result):
            self.rollback("validation_failed")
            return result
        self._state = RepairState.VALIDATED
        self._record("validated", proposal, result)
        return result

    def finalize(self, *, accepted: bool) -> bool:
        proposal = self._require_state(RepairState.VALIDATED)
        if not accepted:
            self.rollback("final_rejected")
            return False
        self._state = RepairState.ACCEPTED
        self._record("accepted", proposal)
        return True

    def rollback(self, reason: str) -> None:
        proposal = self._proposal
        if proposal is None:
            return
        for relative, applied in proposal.files.items():
            target = self._target(relative)
            current = target.read_text(encoding="utf-8") if target.exists() else None
            if current != applied:
                raise RuntimeError("rollback refused because an approved file changed externally: " + relative)
        for relative, original in self._originals.items():
            target = self._target(relative)
            if original is None:
                target.unlink()
                self._remove_empty_parents(target.parent)
            else:
                target.write_text(original, encoding="utf-8")
        self._state = RepairState.ROLLED_BACK
        self._record("rolled_back", proposal, detail=reason)

    def _validate_scope(self, proposal: RepairProposal) -> None:
        for relative in proposal.files:
            target = self._target(relative)
            if target.suffix == ".env" or ".env" in target.name:
                raise ValueError("secret files are outside the supervised repair scope.")

    def _target(self, relative: str) -> Path:
        candidate = (self._root / relative).resolve()
        if candidate == self._root or self._root not in candidate.parents:
            raise ValueError("repair scope must remain inside the project root.")
        return candidate

    def _metrics_improved(self, proposal: RepairProposal, result: RepairValidation) -> bool:
        for name, direction in proposal.metric_directions.items():
            before, after = result.before_metrics.get(name), result.after_metrics.get(name)
            if before is None or after is None:
                return False
            if direction == "decrease" and not after < before:
                return False
            if direction == "increase" and not after > before:
                return False
            if direction == "equal" and after != before:
                return False
        return True

    def _require_state(self, state: RepairState) -> RepairProposal:
        if self._proposal is None or self._state is not state:
            raise RuntimeError("repair is not in the required supervised state.")
        return self._proposal

    def _record(self, event: str, proposal: RepairProposal, result: RepairValidation | None = None, detail: str = "") -> None:
        self.audit_log.append({"event": event, "proposal_id": proposal.proposal_id, "files": tuple(proposal.files), "tests": proposal.focused_tests, "detail": detail or (result.detail if result else "")})

    def _remove_empty_parents(self, path: Path) -> None:
        while path != self._root:
            try:
                path.rmdir()
            except OSError:
                return
            path = path.parent
