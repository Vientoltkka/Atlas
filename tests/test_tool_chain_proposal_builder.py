from __future__ import annotations

from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.tool_chain_proposal_builder import (
    StructuredToolChainProposal,
    StructuredToolChainStepProposal,
    ToolChainProposalBuilder,
    ToolChainProposalError,
    ToolChainProposalStatus,
)
from tools.tool_chain_runner import ToolChainStep
from tools.tool_proposal_builder import ToolProposalStatus


def _decision(
    mode: ExecutionMode = ExecutionMode.TOOL_CHAIN,
    candidates: tuple[str, ...] = ("file.read", "file.write"),
) -> ExecutionDecision:
    return ExecutionDecision(
        mode=mode,
        reason="test decision",
        confidence=0.78,
        candidate_tools=candidates,
    )


def _builder() -> ToolChainProposalBuilder:
    return Bootstrap.build_tool_chain_proposal_builder()


def _proposal(prompt: str) -> StructuredToolChainProposal:
    selector = Bootstrap.build_tool_selector()
    engine = Bootstrap.build_execution_decision_engine(selector)
    decision = engine.decide(prompt)
    return _builder().build(prompt, decision)


def test_read_then_write_chain_is_complete() -> None:
    proposal = _proposal("Lee README.md y copia su contenido en resumen.txt")

    assert proposal.status == ToolChainProposalStatus.COMPLETE
    assert [step.id for step in proposal.steps] == ["read", "write"]
    assert [step.tool_name for step in proposal.steps] == ["file.read", "file.write"]
    assert dict(proposal.steps[0].arguments) == {"path": "README.md"}
    assert dict(proposal.steps[1].arguments) == {
        "path": "resumen.txt",
        "content": "${steps.read.output.content}",
    }
    assert proposal.steps[1].depends_on == ("read",)


def test_list_then_write_chain_reuses_previous_output() -> None:
    proposal = _proposal("Lista la carpeta tools y guarda el resultado en listado.txt")

    assert proposal.status == ToolChainProposalStatus.COMPLETE
    assert [step.tool_name for step in proposal.steps] == ["directory.list", "file.write"]
    assert dict(proposal.steps[0].arguments) == {"path": "tools"}
    assert dict(proposal.steps[1].arguments) == {
        "path": "listado.txt",
        "content": "${steps.list.output}",
    }


def test_three_step_chain_preserves_semantic_order() -> None:
    proposal = _proposal(
        "Lee README.md, despues lista la carpeta tools y luego guarda ambos resultados en resumen.txt"
    )

    assert proposal.status == ToolChainProposalStatus.COMPLETE
    assert [step.id for step in proposal.steps] == ["read", "list", "write"]
    assert [step.tool_name for step in proposal.steps] == [
        "file.read",
        "directory.list",
        "file.write",
    ]
    assert dict(proposal.steps[2].arguments) == {
        "path": "resumen.txt",
        "content": "${steps.read.output.content}\n${steps.list.output}",
    }
    assert set(proposal.steps[2].references) == {
        "${steps.read.output.content}",
        "${steps.list.output}",
    }
    assert proposal.steps[2].depends_on == ("read", "list")


def test_unique_step_ids_for_repeated_actions() -> None:
    proposal = _builder().build(
        "Lee README.md y escribe su contenido dos veces en resumen.txt",
        _decision(candidates=("file.read", "file.write", "file.write")),
    )

    assert [step.id for step in proposal.steps] == ["read", "write", "write_2"]


def test_unregistered_tool_name_is_unsupported() -> None:
    proposal = _builder().build(
        "Borra README.md y crea otro",
        _decision(candidates=("file.delete", "file.write")),
    )

    assert proposal.status == ToolChainProposalStatus.UNSUPPORTED
    assert proposal.steps[0].tool_name == "file.delete"
    assert proposal.steps[0].status == ToolProposalStatus.UNSUPPORTED


def test_step_with_missing_arguments_makes_chain_incomplete() -> None:
    proposal = _builder().build(
        "Lee este archivo y guardalo en resumen.txt",
        _decision(candidates=("file.read", "file.write")),
    )

    assert proposal.status == ToolChainProposalStatus.INCOMPLETE
    assert proposal.missing_information == ("read.path",)


def test_step_with_ambiguous_arguments_makes_chain_ambiguous() -> None:
    proposal = _builder().build(
        "Lee README.md y escribe algo en un archivo",
        _decision(candidates=("file.read", "file.write")),
    )

    assert proposal.status == ToolChainProposalStatus.AMBIGUOUS
    assert set(proposal.ambiguous_information) == {"write.path", "write.content"}


def test_reference_to_missing_step_is_rejected_on_conversion() -> None:
    proposal = StructuredToolChainProposal(
        steps=(
            StructuredToolChainStepProposal(
                id="write",
                tool_name="file.write",
                arguments={
                    "path": "out.txt",
                    "content": "${steps.missing.output.content}",
                },
                references=("${steps.missing.output.content}",),
                status=ToolProposalStatus.COMPLETE,
            ),
        ),
        status=ToolChainProposalStatus.COMPLETE,
    )

    with pytest.raises(ToolChainProposalError, match="does not exist"):
        _builder().to_tool_chain_definition(proposal)


def test_reference_to_future_step_is_rejected_on_conversion() -> None:
    proposal = StructuredToolChainProposal(
        steps=(
            StructuredToolChainStepProposal(
                id="write",
                tool_name="file.write",
                arguments={
                    "path": "out.txt",
                    "content": "${steps.read.output.content}",
                },
                references=("${steps.read.output.content}",),
                status=ToolProposalStatus.COMPLETE,
            ),
            StructuredToolChainStepProposal(
                id="read",
                tool_name="file.read",
                arguments={"path": "README.md"},
                status=ToolProposalStatus.COMPLETE,
            ),
        ),
        status=ToolChainProposalStatus.COMPLETE,
    )

    with pytest.raises(ToolChainProposalError, match="previous step"):
        _builder().to_tool_chain_definition(proposal)


def test_duplicate_step_id_is_rejected_on_conversion() -> None:
    proposal = StructuredToolChainProposal(
        steps=(
            StructuredToolChainStepProposal(
                id="read",
                tool_name="file.read",
                arguments={"path": "README.md"},
                status=ToolProposalStatus.COMPLETE,
            ),
            StructuredToolChainStepProposal(
                id="read",
                tool_name="file.read",
                arguments={"path": "TASKS.md"},
                status=ToolProposalStatus.COMPLETE,
            ),
        ),
        status=ToolChainProposalStatus.COMPLETE,
    )

    with pytest.raises(ToolChainProposalError, match="duplicate"):
        _builder().to_tool_chain_definition(proposal)


def test_direct_response_decision_is_rejected() -> None:
    proposal = _builder().build(
        "Hola",
        _decision(mode=ExecutionMode.DIRECT_RESPONSE, candidates=()),
    )

    assert proposal.status == ToolChainProposalStatus.UNSUPPORTED
    assert proposal.steps == ()


def test_single_tool_decision_is_rejected() -> None:
    proposal = _builder().build(
        "Lee README.md",
        _decision(mode=ExecutionMode.SINGLE_TOOL, candidates=("file.read",)),
    )

    assert proposal.status == ToolChainProposalStatus.UNSUPPORTED
    assert proposal.steps == ()


def test_builder_does_not_execute_tools_or_call_runner() -> None:
    proposal = _proposal("Lee README.md y copia su contenido en resumen.txt")

    assert proposal.status == ToolChainProposalStatus.COMPLETE
    assert all(step.status == ToolProposalStatus.COMPLETE for step in proposal.steps)


def test_builder_does_not_create_or_modify_files(tmp_path) -> None:
    target = tmp_path / "resumen.txt"

    proposal = _builder().build(
        f"Lee README.md y copia su contenido en {target}",
        _decision(),
    )

    assert proposal.status == ToolChainProposalStatus.COMPLETE
    assert target.exists() is False


def test_delete_request_does_not_invent_tool() -> None:
    proposal = _proposal("Borra README.md y crea otro")

    assert proposal.status == ToolChainProposalStatus.UNSUPPORTED
    assert all(step.tool_name != "file.delete" for step in proposal.steps)


def test_unknown_reference_field_is_rejected() -> None:
    proposal = StructuredToolChainProposal(
        steps=(
            StructuredToolChainStepProposal(
                id="read",
                tool_name="file.read",
                arguments={"path": "README.md"},
                status=ToolProposalStatus.COMPLETE,
            ),
            StructuredToolChainStepProposal(
                id="write",
                tool_name="file.write",
                arguments={
                    "path": "out.txt",
                    "content": "${steps.read.output.missing}",
                },
                references=("${steps.read.output.missing}",),
                status=ToolProposalStatus.COMPLETE,
            ),
        ),
        status=ToolChainProposalStatus.COMPLETE,
    )

    with pytest.raises(ToolChainProposalError, match="field is not supported"):
        _builder().to_tool_chain_definition(proposal)


def test_complete_proposal_converts_to_tool_chain_runner_format() -> None:
    proposal = _proposal("Lee README.md y copia su contenido en resumen.txt")

    definition = _builder().to_tool_chain_definition(proposal)

    assert isinstance(definition, tuple)
    assert all(isinstance(step, ToolChainStep) for step in definition)
    assert [(step.step_id, step.tool_name) for step in definition] == [
        ("read", "file.read"),
        ("write", "file.write"),
    ]
    assert dict(definition[1].arguments) == {
        "path": "resumen.txt",
        "content": "${steps.read.output.content}",
    }


def test_non_complete_proposal_does_not_convert() -> None:
    proposal = _builder().build(
        "Lee este archivo y guardalo en resumen.txt",
        _decision(),
    )

    with pytest.raises(ToolChainProposalError):
        _builder().to_tool_chain_definition(proposal)


def test_conversion_revalidates_arguments() -> None:
    proposal = StructuredToolChainProposal(
        steps=(
            StructuredToolChainStepProposal(
                id="read",
                tool_name="file.read",
                arguments={},
                status=ToolProposalStatus.COMPLETE,
            ),
        ),
        status=ToolChainProposalStatus.COMPLETE,
    )

    with pytest.raises(Exception, match="required argument is missing"):
        _builder().to_tool_chain_definition(proposal)


def test_steps_and_arguments_are_immutable() -> None:
    proposal = _proposal("Lee README.md y copia su contenido en resumen.txt")

    with pytest.raises(TypeError):
        proposal.steps[0].arguments["path"] = "other.md"  # type: ignore[index]

    with pytest.raises(AttributeError):
        proposal.steps[0].id = "other"  # type: ignore[misc]


def test_confidence_is_normalized() -> None:
    proposal = _builder().build(
        "Lee README.md y copia su contenido en resumen.txt",
        ExecutionDecision(
            mode=ExecutionMode.TOOL_CHAIN,
            reason="test",
            confidence=2.0,
            candidate_tools=("file.read", "file.write"),
        ),
    )

    assert 0.0 <= proposal.confidence <= 1.0


def test_conversational_plus_tool_chain_is_unsupported_when_forced() -> None:
    proposal = _builder().build(
        "Explicame Git y despues lee README.md",
        _decision(candidates=("direct.response", "file.read")),
    )

    assert proposal.status == ToolChainProposalStatus.UNSUPPORTED
