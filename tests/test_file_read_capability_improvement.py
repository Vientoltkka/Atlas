from __future__ import annotations

from pathlib import Path

import pytest

from core.file_read_capability_builder import FileReadCapabilityImprovementBuilder
from core.self_improvement_conversation import (
    ImprovementClassification,
    ImprovementDiagnosis,
    SelfImprovementConversation,
    SupervisedRepairBuilderRegistry,
)
from core.supervised_repair import RepairProposal, RepairValidation

_ROOT = Path(__file__).resolve().parents[1]

_PROMPT = "Atlas, mejora tu capacidad para trabajar con archivos sin romper las funciones actuales."
_SCOPE = ("tools/filesystem/read_file_tool.py", "bootstrap/bootstrap.py", "tests/test_read_file_tool.py")


class _CapabilityImprovementFixtureBuilder:
    """Fixture builder exercising the supervised capability improvement path."""

    def __init__(self, *, passed: bool = True) -> None:
        self._passed = passed

    def diagnose(self, prompt: str) -> ImprovementDiagnosis | None:
        if "archivos" not in prompt:
            return None
        return ImprovementDiagnosis(
            ImprovementClassification.CAPABILITY_IMPROVEMENT,
            prompt,
            ("capability_source.txt", "tests/test_capability_fixture.py"),
            ("tests/test_capability_fixture.py",),
            ("capacidades_soportadas",),
            "fixture risk",
            "fixture finding",
        )

    def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
        return diagnosis.scope == ("capability_source.txt", "tests/test_capability_fixture.py")

    def build(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if not self.can_handle(diagnosis, _prompt):
            return None
        return RepairProposal(
            "improvement.fixture.bounded-read",
            diagnosis.objective,
            {"capability_source.txt": "improved\n"},
            ("tests/test_capability_fixture.py",),
            {"capacidades_soportadas": "increase"},
        )

    def validator(self, _proposal: RepairProposal) -> RepairValidation:
        return RepairValidation(self._passed, {"capacidades_soportadas": 1.0}, {"capacidades_soportadas": 2.0}, "fixture validated")


def _conversation(tmp_path: Path, *, passed: bool = True) -> SelfImprovementConversation:
    return SelfImprovementConversation(tmp_path, builders=(_CapabilityImprovementFixtureBuilder(passed=passed),))


def test_real_prompt_is_a_capability_improvement_request() -> None:
    assert SelfImprovementConversation.is_self_improvement_request(_PROMPT) is True


def test_real_prompt_is_classified_as_capability_improvement() -> None:
    diagnosis = SelfImprovementConversation(_ROOT).diagnose(_PROMPT)

    assert diagnosis.classification is ImprovementClassification.CAPABILITY_IMPROVEMENT
    assert diagnosis.scope == _SCOPE


def test_registry_resolves_the_file_read_builder() -> None:
    registry = SupervisedRepairBuilderRegistry((FileReadCapabilityImprovementBuilder(_ROOT),))

    diagnosis = FileReadCapabilityImprovementBuilder(_ROOT).diagnose(_PROMPT)

    assert diagnosis is not None
    assert registry.builder_for(diagnosis, _PROMPT) is not None


def test_proposal_is_concrete_with_zero_writes_before_authorization(tmp_path: Path) -> None:
    target = tmp_path / "capability_source.txt"
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(tmp_path)

    response = conversation.handle(_PROMPT)

    assert "improvement.fixture.bounded-read" in response
    assert "mejora una capacidad" in response
    assert "No he modificado nada." in response
    assert target.read_text(encoding="utf-8") == "original\n"


def test_authorize_validate_and_accept_conserves_the_improvement(tmp_path: Path) -> None:
    target = tmp_path / "capability_source.txt"
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(tmp_path)
    conversation.handle(_PROMPT)

    validated = conversation.handle("sí")

    assert "Antes/después: capacidades_soportadas: 1.0 -> 2.0" in validated
    assert target.read_text(encoding="utf-8") == "improved\n"
    assert conversation.handle("sí") == "Reparación aceptada. Se conserva el cambio validado."
    assert target.read_text(encoding="utf-8") == "improved\n"


def test_final_rejection_rolls_back_exact_scope(tmp_path: Path) -> None:
    target, unrelated = tmp_path / "capability_source.txt", tmp_path / "unrelated.txt"
    target.write_text("original\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    conversation = _conversation(tmp_path)
    conversation.handle(_PROMPT)
    conversation.handle("sí")

    assert "restauró exactamente el estado anterior" in conversation.handle("no")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_failed_validation_rolls_back_exact_scope(tmp_path: Path) -> None:
    target = tmp_path / "capability_source.txt"
    target.write_text("original\n", encoding="utf-8")
    conversation = _conversation(tmp_path, passed=False)
    conversation.handle(_PROMPT)

    assert "se restauró exactamente" in conversation.handle("sí")
    assert target.read_text(encoding="utf-8") == "original\n"


def test_out_of_scope_improvement_is_blocked_before_any_write(tmp_path: Path) -> None:
    class _EscalatingBuilder(_CapabilityImprovementFixtureBuilder):
        def build(self, diagnosis: ImprovementDiagnosis, prompt: str) -> RepairProposal | None:
            return RepairProposal(
                "improvement.escalated",
                diagnosis.objective,
                {".env": "SECRET=1\n"},
                (),
                {"capacidades_soportadas": "increase"},
            )

    conversation = SelfImprovementConversation(tmp_path, builders=(_EscalatingBuilder(),))

    with pytest.raises(ValueError):
        conversation.handle(_PROMPT)
    assert not (tmp_path / ".env").exists()


def test_unknown_builder_for_improvement_stops_safely(tmp_path: Path) -> None:
    class _UnresolvableBuilder(_CapabilityImprovementFixtureBuilder):
        def can_handle(self, diagnosis: ImprovementDiagnosis, _prompt: str) -> bool:
            return False

    conversation = SelfImprovementConversation(tmp_path, builders=(_UnresolvableBuilder(),))

    response = conversation.handle(_PROMPT)

    assert response.startswith("CLARIFICATION_REQUIRED")
    assert conversation.proposal is None
    assert not conversation.active


def test_real_builder_prepares_the_bounded_read_proposal_without_writes() -> None:
    source = _ROOT / "tools/filesystem/read_file_tool.py"
    bootstrap = _ROOT / "bootstrap/bootstrap.py"
    test_file = _ROOT / "tests/test_read_file_tool.py"
    before = (source.read_text(encoding="utf-8"), bootstrap.read_text(encoding="utf-8"), test_file.exists())
    builder = FileReadCapabilityImprovementBuilder(_ROOT)
    diagnosis = builder.diagnose(_PROMPT)

    proposal = builder.build(diagnosis, _PROMPT)

    assert proposal is not None
    assert proposal.proposal_id == "improvement.read-file.bounded-limit-read"
    assert set(proposal.files) == set(_SCOPE)
    assert 'ToolParameterSchema("limit", int, minimum=1)' in proposal.files["bootstrap/bootstrap.py"]
    assert "test_read_with_limit_returns_first_lines" in proposal.files["tests/test_read_file_tool.py"]
    assert proposal.metric_directions == {"lecturas_de_archivo_acotadas_correctas": "increase"}
    after = (source.read_text(encoding="utf-8"), bootstrap.read_text(encoding="utf-8"), test_file.exists())
    assert before == after
    assert not builder.can_handle(
        ImprovementDiagnosis(ImprovementClassification.CODE_REPAIR, "x", _SCOPE, (), (), "x", "x"),
        "x",
    )
