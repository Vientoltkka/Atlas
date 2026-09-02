from __future__ import annotations

from pathlib import Path

import pytest

from core.routing_repair_builder import RoutingRepairBuilder
from core.self_improvement_conversation import (
    ImprovementClassification,
    ImprovementDiagnosis,
    SelfImprovementConversation,
    SupervisedRepairBuilderRegistry,
)
from core.supervised_repair import RepairProposal, RepairValidation
from core.voice_repair_builder import VoiceCodeRepairBuilder

_ROOT = Path(__file__).resolve().parents[1]

_DOMAINS = {
    "voice": {
        "prompt": "Atlas, corrige los fallos de la voz sin romper las funciones actuales.",
        "handle_scope": ("use_cases/voice_conversation.py", "tests/test_voice_conversation.py"),
        "source": "voice_source.txt",
        "test": "voice_test.txt",
        "proposal_id": "repair.fixture.voice",
        "real_builder": VoiceCodeRepairBuilder,
        "real_scope": ("use_cases/voice_conversation.py", "tests/test_voice_conversation.py"),
    },
    "routing": {
        "prompt": "Atlas, corrige el fallo de routing sin romper las funciones actuales.",
        "handle_scope": ("core/router.py", "tests/test_router.py"),
        "source": "routing_source.txt",
        "test": "routing_test.txt",
        "proposal_id": "repair.fixture.routing",
        "real_builder": RoutingRepairBuilder,
        "real_scope": ("core/router.py", "tests/test_router.py"),
    },
}


class _FixtureRepairBuilder:
    """Deterministic test-domain builder exercising the shared supervised contract."""

    def __init__(self, handle_scope: tuple[str, ...], fixture_source: str, proposal_id: str, *, passed: bool = True) -> None:
        self._scope, self._source, self._proposal_id, self._passed = handle_scope, fixture_source, proposal_id, passed

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.scope == self._scope

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        return RepairProposal(
            self._proposal_id,
            diagnosis.objective,
            {self._source: "fixed\n"},
            (self._source,),
            {"failures": "decrease"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        return RepairValidation(self._passed, {"failures": 1.0}, {"failures": 0.0}, "fixture validated")


def _conversation(domain: str, tmp_path: Path, *, passed: bool = True) -> SelfImprovementConversation:
    config = _DOMAINS[domain]
    builder = _FixtureRepairBuilder(config["handle_scope"], config["source"], config["proposal_id"], passed=passed)
    return SelfImprovementConversation(tmp_path, builders=(builder,))


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_diagnosis_is_scoped_per_domain(domain: str) -> None:
    diagnosis = SelfImprovementConversation.diagnose(_DOMAINS[domain]["prompt"])

    assert diagnosis.classification is ImprovementClassification.CODE_REPAIR
    assert diagnosis.scope == _DOMAINS[domain]["real_scope"]


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_default_registry_resolves_the_real_builder_for_each_domain(domain: str) -> None:
    registry = SupervisedRepairBuilderRegistry((VoiceCodeRepairBuilder(_ROOT), RoutingRepairBuilder(_ROOT)))
    diagnosis = SelfImprovementConversation.diagnose(_DOMAINS[domain]["prompt"])

    resolved = registry.builder_for(diagnosis, _DOMAINS[domain]["prompt"])

    assert isinstance(resolved, _DOMAINS[domain]["real_builder"])


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_registry_does_not_cross_resolve_between_domains(domain: str) -> None:
    other = next(name for name in _DOMAINS if name != domain)
    registry = SupervisedRepairBuilderRegistry((VoiceCodeRepairBuilder(_ROOT), RoutingRepairBuilder(_ROOT)))

    resolved = registry.builder_for(SelfImprovementConversation.diagnose(_DOMAINS[domain]["prompt"]), _DOMAINS[domain]["prompt"])

    assert isinstance(resolved, _DOMAINS[domain]["real_builder"])
    assert not isinstance(resolved, _DOMAINS[other]["real_builder"])


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_proposal_is_created_without_writes(tmp_path: Path, domain: str) -> None:
    config = _DOMAINS[domain]
    target = tmp_path / config["source"]
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(domain, tmp_path)

    response = conversation.handle(config["prompt"])

    assert config["proposal_id"] in response
    assert "No he modificado nada." in response
    assert "¿Autorizas" in response
    assert target.read_text(encoding="utf-8") == "original\n"
    assert conversation.proposal is not None


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_full_cycle_authorize_validate_and_accept(tmp_path: Path, domain: str) -> None:
    config = _DOMAINS[domain]
    target = tmp_path / config["source"]
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(domain, tmp_path)
    conversation.handle(config["prompt"])

    assert "Antes/después: failures: 1.0 -> 0.0" in conversation.handle("sí")
    assert target.read_text(encoding="utf-8") == "fixed\n"
    assert conversation.handle("sí") == "Reparación aceptada. Se conserva el cambio validado."
    assert target.read_text(encoding="utf-8") == "fixed\n"


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_final_rejection_rolls_back_exact_scope(tmp_path: Path, domain: str) -> None:
    config = _DOMAINS[domain]
    target, unrelated = tmp_path / config["source"], tmp_path / "unrelated.txt"
    target.write_text("original\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    conversation = _conversation(domain, tmp_path)
    conversation.handle(config["prompt"])
    conversation.handle("sí")

    assert "restauró exactamente el estado anterior" in conversation.handle("no")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_failed_validation_rolls_back_exact_scope(tmp_path: Path, domain: str) -> None:
    config = _DOMAINS[domain]
    target = tmp_path / config["source"]
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(domain, tmp_path, passed=False)
    conversation.handle(config["prompt"])

    assert "se restauró exactamente" in conversation.handle("sí")
    assert target.read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize("domain", ["voice", "routing"])
@pytest.mark.parametrize("violation", ["secret", "outside"])
def test_out_of_scope_proposals_are_blocked_before_any_write(tmp_path: Path, domain: str, violation: str) -> None:
    config = _DOMAINS[domain]

    class _EscalatingBuilder(_FixtureRepairBuilder):
        def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None:
            files = {".env": "SECRET=1\n"} if violation == "secret" else {"../outside.txt": "x\n"}
            return RepairProposal("repair.escalated", diagnosis.objective, files, (), {"failures": "decrease"})

    conversation = SelfImprovementConversation(
        tmp_path, builders=(_EscalatingBuilder(config["handle_scope"], config["source"], "repair.fixture"),)
    )

    with pytest.raises(ValueError):
        conversation.handle(config["prompt"])
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / ".env").exists()


def test_unknown_builder_scope_stops_safely_without_changes(tmp_path: Path) -> None:
    conversation = _conversation("voice", tmp_path)

    response = conversation.handle(_DOMAINS["routing"]["prompt"])

    assert response.startswith("CLARIFICATION_REQUIRED")
    assert conversation.proposal is None
    assert not conversation.active


@pytest.mark.parametrize("domain", ["voice", "routing"])
def test_real_builder_prepares_only_the_reviewed_domain_repair(domain: str) -> None:
    config = _DOMAINS[domain]
    builder = config["real_builder"](_ROOT)
    diagnosis = SelfImprovementConversation.diagnose(config["prompt"])

    proposal = builder.build(diagnosis, config["prompt"])

    assert proposal is not None
    assert set(proposal.files) == set(config["real_scope"])
    assert builder.can_handle(diagnosis, config["prompt"])
    assert not builder.can_handle(
        ImprovementDiagnosis(
            ImprovementClassification.CODE_REPAIR,
            "x",
            (*config["real_scope"][:1], "outside.py"),
            (), (), "x", "x",
        ),
        "x",
    )
