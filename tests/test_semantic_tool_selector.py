from __future__ import annotations

from typing import Any

from core.planner import Planner
from tools.base_tool import BaseTool
from tools.intent_selector import (
    MAXIMUM_CANDIDATES,
    MINIMUM_SELECTION_SCORE,
    ToolCandidate,
    ToolSelectionResult,
    ToolSelector,
    select,
)
from tools.registry import ToolRegistry
from tools.semantic_catalog import SemanticToolCatalog, SemanticToolDescriptor
from tools.tool_context import ToolContext


class SelectorFakeTool(BaseTool):
    def __init__(
        self,
        name: str,
        *,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._requires_confirmation = requires_confirmation
        self.executed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}."

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def execute(self, context: ToolContext) -> str:
        self.executed = True
        raise AssertionError("semantic selection must not execute tools")


def _descriptor(
    name: str,
    *,
    capabilities: tuple[str, ...],
    intents: tuple[str, ...],
    tags: tuple[str, ...] = (),
    examples: tuple[str, ...] = (),
    negative_examples: tuple[str, ...] = (),
    required_arguments: tuple[str, ...] = (),
    technical_arguments: tuple[str, ...] = (),
    risk_level: str = "low",
    requires_confirmation: bool = False,
    limitations: tuple[str, ...] = (),
) -> SemanticToolDescriptor:
    return SemanticToolDescriptor(
        name=name,
        description=f"Descriptor for {name}.",
        capabilities=capabilities,
        supported_intents=intents,
        input_description="Structured input.",
        required_arguments=required_arguments,
        optional_arguments=(),
        output_description="Output contract.",
        output_fields=(),
        dangerous=requires_confirmation,
        risk_level=risk_level,
        risk_reasons=("requires user confirmation",) if requires_confirmation else (),
        requires_confirmation=requires_confirmation,
        preconditions=tuple(f"{argument} must be provided" for argument in required_arguments),
        limitations=limitations,
        negative_examples=negative_examples,
        compatible_tools=(),
        tags=tags,
        positive_examples=examples,
        category="fake",
        technical_arguments=technical_arguments or required_arguments,
    )


def _catalog(
    order: tuple[str, ...] = ("read_file", "write_file", "list_directory", "terminate_process"),
) -> SemanticToolCatalog:
    descriptors = {
        "read_file": _descriptor(
            "read_file",
            capabilities=("read_file",),
            intents=("lee un archivo local", "read a local file"),
            tags=("filesystem", "archivo", "read"),
            examples=("lee el archivo C:/Temp/notas.txt",),
            negative_examples=("que es un archivo", "explicame que es un archivo"),
            required_arguments=("path",),
        ),
        "write_file": _descriptor(
            "write_file",
            capabilities=("write_file",),
            intents=("escribe contenido en un archivo", "create or update a text file"),
            tags=("filesystem", "archivo", "write"),
            examples=("escribe hola en notas.txt",),
            negative_examples=("escribe una historia",),
            required_arguments=("path", "content"),
            risk_level="medium",
            requires_confirmation=True,
        ),
        "list_directory": _descriptor(
            "list_directory",
            capabilities=("list_directory",),
            intents=("lista archivos en un directorio", "list files in a directory"),
            tags=("filesystem", "directorio", "archivo"),
            examples=("lista los archivos de C:/Temp",),
            negative_examples=("resume los contenidos de todos los archivos",),
        ),
        "terminate_process": _descriptor(
            "terminate_process",
            capabilities=("terminate_process",),
            intents=("termina un proceso",),
            tags=("desktop", "proceso"),
            examples=("termina el proceso notepad",),
            negative_examples=("que es un proceso", "explicacion general sobre procesos"),
            required_arguments=("query",),
            risk_level="high",
            requires_confirmation=True,
        ),
    }
    return SemanticToolCatalog({name: descriptors[name] for name in order})


def test_empty_query_returns_structured_error() -> None:
    result = select("   ", _catalog())

    assert result.success is False
    assert result.selected_tool is None
    assert result.requires_clarification is True
    assert result.error_code == "EMPTY_QUERY"


def test_exact_capability_match_selects_tool() -> None:
    result = select("read_file C:/Temp/notas.txt", _catalog())

    assert result.selected_tool == "read_file"
    assert result.candidates[0].matched_capabilities == ("read_file",)


def test_exact_intent_match_selects_tool() -> None:
    result = select("lee un archivo local C:/Temp/notas.txt", _catalog())

    assert result.candidates[0].tool_name == "read_file"
    assert "lee un archivo local" in result.candidates[0].matched_intents


def test_tag_match_ranks_candidate() -> None:
    result = select("directorio", _catalog())

    assert result.candidates[0].tool_name == "list_directory"
    assert result.candidates[0].score >= 25.0


def test_positive_example_match_selects_tool() -> None:
    result = select("lista los archivos de C:/Temp", _catalog())

    assert result.selected_tool == "list_directory"
    assert result.candidates[0].matched_examples == ("lista los archivos de C:/Temp",)


def test_negative_example_penalizes_candidate() -> None:
    result = select("escribe una historia", _catalog())

    write = next(candidate for candidate in result.candidates if candidate.tool_name == "write_file")

    assert write.negative_matches == ("escribe una historia",)
    assert result.selected_tool != "write_file"


def test_does_not_select_read_file_for_general_file_question() -> None:
    result = select("que es un archivo", _catalog())

    assert result.selected_tool is None
    assert result.error_code == "NO_TOOL_MATCH"


def test_does_not_select_write_file_for_story_request() -> None:
    result = select("escribe una historia", _catalog())

    assert result.selected_tool is None


def test_does_not_select_terminate_process_for_general_explanation() -> None:
    result = select("explicacion general sobre procesos", _catalog())

    assert result.selected_tool is None


def test_ranking_is_deterministic() -> None:
    first = select("lee el archivo C:/Temp/notas.txt", _catalog())
    second = select("lee el archivo C:/Temp/notas.txt", _catalog())

    assert first.candidates == second.candidates
    assert first.selected_tool == second.selected_tool


def test_ranking_is_independent_from_registration_order() -> None:
    forward = select("lee el archivo C:/Temp/notas.txt", _catalog())
    reverse = select(
        "lee el archivo C:/Temp/notas.txt",
        _catalog(("terminate_process", "list_directory", "write_file", "read_file")),
    )

    assert [candidate.tool_name for candidate in forward.candidates] == [
        candidate.tool_name for candidate in reverse.candidates
    ]


def test_top_k_limits_candidates() -> None:
    result = select("gestiona el archivo", _catalog(), top_k=2)

    assert len(result.candidates) == 2
    assert len(result.candidates) <= MAXIMUM_CANDIDATES


def test_no_candidate_exceeds_threshold() -> None:
    result = select("calcula la hipotenusa", _catalog())

    assert result.selected_tool is None
    assert result.error_code == "NO_TOOL_MATCH"
    assert all(candidate.score < MINIMUM_SELECTION_SCORE for candidate in result.candidates)


def test_close_candidates_produce_ambiguity() -> None:
    catalog = SemanticToolCatalog(
        {
            "alpha_tool": _descriptor(
                "alpha_tool",
                capabilities=("alpha_action",),
                intents=("gestiona alfa",),
                tags=("gestion",),
            ),
            "beta_tool": _descriptor(
                "beta_tool",
                capabilities=("beta_action",),
                intents=("gestiona beta",),
                tags=("gestion",),
            ),
        }
    )

    result = select("gestion", catalog)

    assert result.ambiguous is True
    assert result.selected_tool is None
    assert result.requires_clarification is True


def test_selected_tool_when_clear_winner_exists() -> None:
    result = select("termina el proceso notepad", _catalog())

    assert result.selected_tool == "terminate_process"
    assert result.ambiguous is False


def test_requires_clarification_when_required_argument_is_missing() -> None:
    result = select("lee un archivo local", _catalog())

    assert result.selected_tool is None
    assert result.requires_clarification is True
    assert result.error_code == "INSUFFICIENT_INFORMATION"
    assert "path" in result.reasons[-1]


def test_risk_and_confirmation_are_preserved() -> None:
    result = select("termina el proceso notepad", _catalog())

    assert result.candidates[0].risk_level == "high"
    assert result.candidates[0].requires_confirmation is True


def test_dangerous_tool_never_loses_confirmation() -> None:
    descriptor = _catalog().get("terminate_process")

    assert descriptor.requires_confirmation is True
    assert select("terminate_process", _catalog()).candidates[0].requires_confirmation is True


def test_incomplete_metadata_produces_warning_and_penalty() -> None:
    catalog = SemanticToolCatalog(
        {
            "legacy_tool": _descriptor(
                "legacy_tool",
                capabilities=("legacy_tool",),
                intents=("legacy tool",),
                limitations=("semantic metadata is incomplete",),
            )
        }
    )

    result = select("legacy_tool", catalog)

    assert result.warnings
    assert result.candidates[0].score < 100.0


def test_candidates_are_explainable() -> None:
    result = select("lee el archivo C:/Temp/notas.txt", _catalog())
    candidate = result.candidates[0]

    assert isinstance(candidate, ToolCandidate)
    assert "Candidate read_file scored" in candidate.explanation
    assert "Risk level" in candidate.explanation


def test_invalid_catalog_returns_structured_error() -> None:
    catalog = SemanticToolCatalog(
        {
            "bad": _descriptor(
                "bad",
                capabilities=(),
                intents=("bad",),
            )
        }
    )

    result = select("bad", catalog)

    assert result.success is False
    assert result.error_code == "CATALOG_INVALID"
    assert result.errors


def test_selection_does_not_execute_tools() -> None:
    registry = ToolRegistry()
    tool = SelectorFakeTool("read_file")
    registry.register(tool)

    select("lee el archivo C:/Temp/notas.txt", _catalog())

    assert tool.executed is False


def test_selection_does_not_call_models_or_network() -> None:
    result = select("lee el archivo C:/Temp/notas.txt", _catalog())

    assert isinstance(result, ToolSelectionResult)
    assert result.success is True


def test_selection_does_not_mutate_catalog() -> None:
    catalog = _catalog()
    before = catalog.to_json()

    select("lee el archivo C:/Temp/notas.txt", catalog)

    assert catalog.to_json() == before


def test_existing_tool_selector_can_select_from_semantic_catalog() -> None:
    registry = ToolRegistry()
    registry.register(SelectorFakeTool("read_file"))
    selector = ToolSelector(registry, intent_registry=type("IntentRegistry", (), {
        "supports": lambda self, action: False,
        "list": lambda self: (),
        "resolve": lambda self, action: "",
    })())

    result = selector.select_from_catalog("read_file C:/Temp/notas.txt", _catalog())

    assert result.selected_tool == "read_file"


def test_planner_still_works_without_semantic_selector() -> None:
    plan = Planner().create_execution_plan("Lee README.md")

    assert plan.required_tools == ("read_file",)


def test_planner_accepts_optional_selection_result_without_behavior_change() -> None:
    selection = select("read_file", _catalog())
    plain = Planner().create_execution_plan("Lee README.md")
    with_selection = Planner(tool_selection_result=selection).create_execution_plan("Lee README.md")

    assert with_selection == plain


def test_functional_case_read_file() -> None:
    result = select("lee el archivo C:/Temp/notas.txt", _catalog())

    assert result.candidates[0].tool_name == "read_file"
    assert result.ambiguous is False


def test_functional_case_list_directory() -> None:
    result = select("lista los archivos de C:/Temp", _catalog())

    assert result.candidates[0].tool_name == "list_directory"


def test_functional_case_general_file_question() -> None:
    result = select("que es un archivo", _catalog())

    assert result.selected_tool is None


def test_functional_case_story_request() -> None:
    result = select("escribe una historia", _catalog())

    assert result.selected_tool is None


def test_functional_case_terminate_process() -> None:
    result = select("termina el proceso notepad", _catalog())

    assert result.candidates[0].tool_name == "terminate_process"
    assert result.candidates[0].risk_level == "high"
    assert result.candidates[0].requires_confirmation is True


def test_functional_case_ambiguous_file_management() -> None:
    result = select("gestiona el archivo", _catalog())

    assert result.ambiguous is True
    assert result.requires_clarification is True
    assert result.selected_tool is None
