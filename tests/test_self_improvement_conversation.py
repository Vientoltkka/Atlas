from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.orchestrator import AtlasOrchestrator
from core.router import Router
from core.self_improvement_conversation import ImprovementDiagnosis, SelfImprovementConversation
from core.supervised_repair import ImprovementClassification, RepairProposal, RepairValidation
from memory.conversation import ConversationMemory


class _Chat:
    name = "chat"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, model, messages):
        self.calls += 1
        return "ruta normal"


class _Models:
    def choose_model(self, _task):
        return "test"


def _conversation(root: Path, *, passed: bool = True) -> SelfImprovementConversation:
    def build(diagnosis: ImprovementDiagnosis, _prompt: str) -> RepairProposal | None:
        if diagnosis.classification is not ImprovementClassification.CODE_REPAIR:
            return None
        return RepairProposal(
            "repair.dialogue-fixture",
            diagnosis.objective,
            {"fixture.txt": "fixed\n"},
            ("tests/test_self_improvement_conversation.py",),
            {"failures": "decrease"},
        )

    return SelfImprovementConversation(
        root,
        proposal_builder=build,
        validator_factory=lambda _proposal: lambda _: RepairValidation(passed, {"failures": 1}, {"failures": 0}, "fixture validated"),
    )


def _orchestrator(root: Path, *, passed: bool = True):
    chat = _Chat()
    app = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=_Models(),
        memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda name: chat if name == "chat" else None),
        write_file=SimpleNamespace(execute=lambda *_: None),
        project_root=root,
        self_improvement_conversation=_conversation(root, passed=passed),
    )
    return app, chat


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Atlas, corrige los fallos de la voz", True),
        ("Atlas, optimiza tu voz", True),
        ("Haz que puedas organizar mis tareas", True),
        ("Crea la capacidad para que Atlas lea calendarios", True),
        ("corrige este texto", False),
        ("mejora esta carta", False),
    ],
)
def test_detects_only_atlas_self_improvement_intent(prompt: str, expected: bool) -> None:
    assert SelfImprovementConversation.is_self_improvement_request(prompt) is expected


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Atlas usa una capacidad existente", ImprovementClassification.REUSE),
        ("Atlas crea una skill para algo", ImprovementClassification.SKILL_GAP),
        ("Atlas corrige los fallos de la voz", ImprovementClassification.CODE_REPAIR),
        ("Atlas crea la capacidad para un proveedor externo", ImprovementClassification.CAPABILITY_GAP),
        ("Atlas mejora", ImprovementClassification.CLARIFICATION_REQUIRED),
    ],
)
def test_classifies_self_improvement_requests(prompt: str, expected: ImprovementClassification) -> None:
    assert SelfImprovementConversation.diagnose(prompt).classification is expected


def test_voice_diagnosis_is_limited_to_voice_scope() -> None:
    diagnosis = SelfImprovementConversation.diagnose("Atlas, corrige los fallos de la voz")
    assert diagnosis.scope == ("use_cases/voice_conversation.py", "use_cases/speech_engine.py", "tests/test_voice_conversation.py")


def test_e2e_propose_authorize_validate_and_accept(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    app, _ = _orchestrator(tmp_path)

    proposal = app.process_prompt("Atlas, corrige los fallos de la voz", confirm=lambda _: "")
    assert "proposal_id: repair.dialogue-fixture" in proposal
    assert target.read_text(encoding="utf-8") == "original\n"
    validated = app.process_prompt("sí", confirm=lambda _: "")
    assert "Antes/después: failures: 1 -> 0" in validated
    assert target.read_text(encoding="utf-8") == "fixed\n"
    assert app.process_prompt("sí", confirm=lambda _: "") == "Reparación aceptada. Se conserva el cambio validado."
    assert target.read_text(encoding="utf-8") == "fixed\n"


def test_rejection_rolls_back_exact_fixture_scope(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    unrelated = tmp_path / "unrelated.txt"
    target.write_text("original\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    app, _ = _orchestrator(tmp_path)
    app.process_prompt("Atlas, corrige los fallos de la voz", confirm=lambda _: "")
    app.process_prompt("sí", confirm=lambda _: "")

    assert "restauró exactamente" in app.process_prompt("no", confirm=lambda _: "")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_validation_failure_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    app, _ = _orchestrator(tmp_path, passed=False)
    app.process_prompt("Atlas, corrige los fallos de la voz", confirm=lambda _: "")

    assert "se restauró exactamente" in app.process_prompt("sí", confirm=lambda _: "")
    assert target.read_text(encoding="utf-8") == "original\n"


def test_capability_gap_stops_without_changes(tmp_path: Path) -> None:
    app, _ = _orchestrator(tmp_path)
    response = app.process_prompt("Atlas, crea la capacidad para un proveedor externo", confirm=lambda _: "")
    assert response.startswith("CAPABILITY_GAP")
    assert not (tmp_path / "fixture.txt").exists()


def test_normal_conversation_is_not_captured(tmp_path: Path) -> None:
    app, chat = _orchestrator(tmp_path)
    assert app.process_prompt("corrige este texto", confirm=lambda _: "") == "ruta normal"
    assert chat.calls == 1
