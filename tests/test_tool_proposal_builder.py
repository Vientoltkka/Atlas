from __future__ import annotations

from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import ArgumentValidator
from tools.base_tool import BaseTool
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.tool_context import ToolContext
from tools.tool_proposal_builder import (
    StructuredToolProposal,
    ToolProposalBuilder,
    ToolProposalError,
    ToolProposalStatus,
)


class ExplodingTool(BaseTool):
    def __init__(self) -> None:
        self.executed = False

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Must not be executed."

    def execute(
        self,
        context: ToolContext,
    ) -> Any:
        self.executed = True
        raise AssertionError("proposal building must not execute tools")


def _decision(
    mode: ExecutionMode = ExecutionMode.SINGLE_TOOL,
    candidates: tuple[str, ...] = ("file.read",),
) -> ExecutionDecision:
    return ExecutionDecision(
        mode=mode,
        reason="test decision",
        confidence=0.82,
        candidate_tools=candidates,
    )


def _builder() -> ToolProposalBuilder:
    return Bootstrap.build_tool_proposal_builder()


def _proposal(prompt: str) -> StructuredToolProposal:
    selector = Bootstrap.build_tool_selector()
    engine = Bootstrap.build_execution_decision_engine(selector)
    return _builder().build(prompt, engine.decide(prompt))


def test_read_file_with_complete_path_builds_complete_proposal() -> None:
    proposal = _proposal("Lee el archivo README.md")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert proposal.tool_name == "file.read"
    assert dict(proposal.arguments) == {"path": "README.md"}
    assert proposal.missing_arguments == ()
    assert proposal.ambiguous_arguments == ()
    assert proposal.executable is True


def test_list_directory_with_path_builds_complete_proposal() -> None:
    proposal = _proposal("Lista los archivos de la carpeta tools")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert proposal.tool_name == "directory.list"
    assert dict(proposal.arguments) == {"path": "tools"}


def test_calendar_request_builds_complete_normalized_proposal() -> None:
    proposal = _proposal(
        "Lista eventos del calendario entre "
        "2026-08-09T09:00:00+01:00 y 2026-08-09T10:00:00+01:00 "
        "max_results=3"
    )

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert proposal.tool_name == "calendar.events.list"
    assert dict(proposal.arguments) == {
        "time_min": "2026-08-09T09:00:00+01:00",
        "time_max": "2026-08-09T10:00:00+01:00",
        "max_results": 3,
    }


def test_calendar_request_without_range_requires_clarification() -> None:
    proposal = _proposal("Lista eventos del calendario")

    assert proposal.status == ToolProposalStatus.INCOMPLETE
    assert proposal.tool_name == "calendar.events.list"
    assert proposal.missing_arguments == ("time_min", "time_max")
    assert proposal.executable is False


def test_calendar_request_without_rfc3339_timezone_requires_clarification() -> None:
    proposal = _proposal(
        "Lista eventos del calendario entre "
        "2026-08-09T09:00:00 y 2026-08-09T10:00:00"
    )

    assert proposal.status == ToolProposalStatus.INCOMPLETE
    assert proposal.missing_arguments == ("time_min", "time_max")


def test_write_file_with_path_and_content_builds_complete_proposal() -> None:
    proposal = _proposal("Escribe hola en prueba.txt")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert proposal.tool_name == "file.write"
    assert dict(proposal.arguments) == {"path": "prueba.txt", "content": "hola"}


def test_read_file_without_path_returns_incomplete_proposal() -> None:
    proposal = _proposal("Lee este archivo")

    assert proposal.status == ToolProposalStatus.INCOMPLETE
    assert proposal.tool_name == "file.read"
    assert dict(proposal.arguments) == {}
    assert proposal.missing_arguments == ("path",)
    assert proposal.executable is False


def test_write_file_without_content_returns_incomplete_proposal() -> None:
    proposal = _proposal("Escribe en prueba.txt")

    assert proposal.status == ToolProposalStatus.INCOMPLETE
    assert proposal.tool_name == "file.write"
    assert dict(proposal.arguments) == {"path": "prueba.txt"}
    assert proposal.missing_arguments == ("content",)


def test_write_file_without_path_marks_path_ambiguous() -> None:
    proposal = _proposal("Escribe hola en un archivo")

    assert proposal.status == ToolProposalStatus.AMBIGUOUS
    assert proposal.tool_name == "file.write"
    assert dict(proposal.arguments) == {"content": "hola"}
    assert proposal.ambiguous_arguments == ("path",)
    assert proposal.executable is False


def test_ambiguous_request_marks_ambiguous_content_and_target() -> None:
    proposal = _proposal("Escribe algo en un archivo")

    assert proposal.status == ToolProposalStatus.AMBIGUOUS
    assert proposal.tool_name == "file.write"
    assert dict(proposal.arguments) == {}
    assert set(proposal.ambiguous_arguments) == {"path", "content"}


def test_unregistered_candidate_is_rejected() -> None:
    proposal = _builder().build(
        "Borra README.md",
        _decision(candidates=("file.delete",)),
    )

    assert proposal.status == ToolProposalStatus.UNSUPPORTED
    assert proposal.tool_name == "file.delete"
    assert proposal.executable is False


def test_delete_request_does_not_invent_file_delete_tool() -> None:
    proposal = _proposal("Borra README.md")

    assert proposal.status == ToolProposalStatus.UNSUPPORTED
    assert proposal.tool_name is None
    assert dict(proposal.arguments) == {}


def test_builder_only_uses_decision_candidates() -> None:
    proposal = _builder().build(
        "Lee el archivo README.md",
        _decision(candidates=("directory.list",)),
    )

    assert proposal.tool_name == "directory.list"
    assert proposal.tool_name != "file.read"


def test_direct_response_decision_is_rejected() -> None:
    proposal = _builder().build(
        "Hola",
        _decision(mode=ExecutionMode.DIRECT_RESPONSE, candidates=()),
    )

    assert proposal.status == ToolProposalStatus.UNSUPPORTED
    assert proposal.executable is False


def test_tool_chain_decision_is_rejected() -> None:
    proposal = _builder().build(
        "Lee README.md y copia su contenido en resumen.txt",
        _decision(
            mode=ExecutionMode.TOOL_CHAIN,
            candidates=("file.read", "file.write"),
        ),
    )

    assert proposal.status == ToolProposalStatus.UNSUPPORTED
    assert proposal.executable is False


def test_builder_does_not_execute_tools() -> None:
    tool = ExplodingTool()
    registry = ToolRegistry()
    registry.register(tool)
    intent_registry = ToolIntentRegistry()
    intent_registry.register("file.read", "read_file")
    schema_registry = Bootstrap.build_argument_schema_registry()
    selector = ToolSelector(registry, intent_registry)
    builder = ToolProposalBuilder(
        registry,
        selector,
        schema_registry,
        ArgumentValidator(schema_registry),
    )

    proposal = builder.build("Lee README.md", _decision())

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert tool.executed is False


def test_builder_does_not_create_files(tmp_path) -> None:
    target = tmp_path / "prueba.txt"

    proposal = _proposal(f"Escribe hola en {target}")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert target.exists() is False


def test_confidence_is_always_normalized() -> None:
    proposal = _builder().build(
        "Lee README.md",
        ExecutionDecision(
            mode=ExecutionMode.SINGLE_TOOL,
            reason="test",
            confidence=2.0,
            candidate_tools=("file.read",),
        ),
    )

    assert 0.0 <= proposal.confidence <= 1.0


def test_complete_proposal_converts_to_tool_intent() -> None:
    proposal = _proposal("Lee el archivo README.md")

    intent = proposal.to_tool_intent(
        Bootstrap.build_tool_selector(),
        Bootstrap.build_argument_validator(),
    )

    assert intent.action == "file.read"
    assert dict(intent.arguments) == {"path": "README.md"}


def test_builder_converts_complete_proposal_after_revalidation() -> None:
    proposal = _proposal("Lee el archivo README.md")

    intent = _builder().to_tool_intent(proposal)

    assert intent.action == "file.read"
    assert dict(intent.arguments) == {"path": "README.md"}


def test_incomplete_proposal_does_not_convert_to_tool_intent() -> None:
    proposal = _proposal("Lee este archivo")

    with pytest.raises(ToolProposalError):
        proposal.to_tool_intent(
            Bootstrap.build_tool_selector(),
            Bootstrap.build_argument_validator(),
        )


def test_argument_validator_remains_final_validation_source() -> None:
    proposal = _builder().build(
        "Lee README.md",
        _decision(candidates=("file.read",)),
        candidate_tools=("file.read",),
    )

    selector = Bootstrap.build_tool_selector()
    validator = Bootstrap.build_argument_validator()
    validation = validator.validate(
        selector.select(proposal.to_tool_intent(selector, validator))
    )

    assert validation.valid is True
    assert dict(validation.validated_arguments) == {"path": "README.md"}


def test_quoted_path_with_spaces_is_preserved() -> None:
    proposal = _proposal('Lee el archivo "Mis notas.md"')

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert dict(proposal.arguments) == {"path": "Mis notas.md"}


def test_windows_path_is_preserved() -> None:
    proposal = _proposal(r"Lee el archivo C:\AI\Atlas\README.md")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert dict(proposal.arguments) == {"path": r"C:\AI\Atlas\README.md"}


def test_directory_without_quotes_does_not_capture_trailing_words() -> None:
    proposal = _proposal("Lista los archivos de la carpeta tools por favor")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert dict(proposal.arguments) == {"path": "tools"}


def test_write_content_preserves_spaces_quotes_and_unicode() -> None:
    proposal = _proposal('Escribe "hola, Víctor" en prueba.txt')

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert dict(proposal.arguments) == {
        "path": "prueba.txt",
        "content": "hola, Víctor",
    }


def test_proposal_arguments_are_immutable_after_build() -> None:
    proposal = _proposal("Lee el archivo README.md")

    with pytest.raises(TypeError):
        proposal.arguments["path"] = "other.md"  # type: ignore[index]


def test_nested_argument_values_are_immutable_after_build() -> None:
    proposal = _proposal("Pulsa Ctrl+S")

    assert dict(proposal.arguments) == {"keys": ("ctrl", "s")}


def test_desktop_open_application_extracts_registered_tool_arguments() -> None:
    proposal = _proposal("Abre VS Code")

    assert proposal.status == ToolProposalStatus.COMPLETE
    assert proposal.tool_name == "desktop.application.open"
    assert dict(proposal.arguments) == {"application": "VS Code"}


def test_hotkey_extracts_keys_but_keeps_missing_window_title() -> None:
    proposal = _proposal("Pulsa Ctrl+S")

    assert proposal.status == ToolProposalStatus.INCOMPLETE
    assert proposal.tool_name == "desktop.hotkey.press"
    assert dict(proposal.arguments) == {"keys": ("ctrl", "s")}
    assert proposal.missing_arguments == ("window_title",)
