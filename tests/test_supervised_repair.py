from __future__ import annotations

from pathlib import Path

import pytest

from core.supervised_repair import (
    ImprovementClassification,
    RepairProposal,
    RepairState,
    RepairValidation,
    SupervisedRepairWorkflow,
)


def _proposal() -> RepairProposal:
    return RepairProposal(
        proposal_id="repair.safe-fixture",
        objective="Repair the isolated fixture.",
        files={"fixture.txt": "fixed\n"},
        focused_tests=("tests/test_supervised_repair.py",),
        metric_directions={"failures": "decrease"},
    )


def _workflow(root: Path, *, passed: bool = True) -> SupervisedRepairWorkflow:
    return SupervisedRepairWorkflow(
        root,
        validator=lambda _: RepairValidation(passed, {"failures": 1}, {"failures": 0}, "fixture validation"),
    )


@pytest.mark.parametrize(
    ("prompt", "reusable", "expected"),
    [
        ("usa la capacidad existente", True, ImprovementClassification.REUSE),
        ("crea una skill para combinar tools", False, ImprovementClassification.SKILL_GAP),
        ("corrige los fallos de la voz", False, ImprovementClassification.CODE_REPAIR),
        ("falta un proveedor de voz", False, ImprovementClassification.CAPABILITY_GAP),
        ("mejorate entero", False, ImprovementClassification.CLARIFICATION_REQUIRED),
    ],
)
def test_classifies_supervised_improvement_requests(prompt: str, reusable: bool, expected: ImprovementClassification) -> None:
    assert SupervisedRepairWorkflow.classify(prompt, reusable=reusable) is expected


def test_no_change_before_exact_proposal_authorization(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    workflow = _workflow(tmp_path)
    proposal = workflow.propose(_proposal())

    assert target.read_text(encoding="utf-8") == "original\n"
    assert not workflow.authorize_and_apply("AUTORIZAR repair.other invalid")
    assert target.read_text(encoding="utf-8") == "original\n"
    assert workflow.authorize_and_apply(proposal.authorization)
    assert target.read_text(encoding="utf-8") == "fixed\n"
    assert not workflow.authorize_and_apply(proposal.authorization)


def test_validation_records_metrics_and_final_acceptance_preserves_change(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    workflow = _workflow(tmp_path)
    proposal = workflow.propose(_proposal())
    assert workflow.authorize_and_apply(proposal.authorization)

    validation = workflow.validate()

    assert validation.passed
    assert workflow.state is RepairState.VALIDATED
    assert workflow.finalize(accepted=True)
    assert workflow.state is RepairState.ACCEPTED
    assert target.read_text(encoding="utf-8") == "fixed\n"
    assert [entry["event"] for entry in workflow.audit_log] == ["proposed", "applied", "validated", "accepted"]


def test_failed_validation_rolls_back_exact_scope_and_leaves_unrelated_file(tmp_path: Path) -> None:
    target, unrelated = tmp_path / "fixture.txt", tmp_path / "unrelated.txt"
    target.write_text("original\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    workflow = _workflow(tmp_path, passed=False)
    proposal = workflow.propose(_proposal())
    assert workflow.authorize_and_apply(proposal.authorization)

    workflow.validate()

    assert workflow.state is RepairState.ROLLED_BACK
    assert target.read_text(encoding="utf-8") == "original\n"
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_final_rejection_rolls_back_and_preserves_prior_state(tmp_path: Path) -> None:
    target = tmp_path / "fixture.txt"
    target.write_text("original\n", encoding="utf-8")
    workflow = _workflow(tmp_path)
    proposal = workflow.propose(_proposal())
    assert workflow.authorize_and_apply(proposal.authorization)
    workflow.validate()

    assert not workflow.finalize(accepted=False)
    assert workflow.state is RepairState.ROLLED_BACK
    assert target.read_text(encoding="utf-8") == "original\n"


def test_rejects_secret_and_outside_scope_before_snapshot(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    with pytest.raises(ValueError):
        workflow.propose(RepairProposal("repair.secret", "x", {".env": "x"}, ("test",)))
    with pytest.raises(ValueError):
        workflow.propose(RepairProposal("repair.outside", "x", {"../outside.txt": "x"}, ("test",)))
