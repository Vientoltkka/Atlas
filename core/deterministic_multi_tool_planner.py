"""Deterministic multi-tool planning for explicit Atlas patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Callable
import unicodedata

from core.acceptance_criteria import (
    AcceptanceCriterion,
    AcceptanceCriterionKind,
)
from core.execution_plan_validator import ExecutionPlanValidator
from core.planner import ExecutionPlan, ExecutionStep
from core.step_output_reference import StepOutputReference
from tools.intent_selector import ToolSelector
from tools.semantic_catalog import SemanticToolCatalog, SemanticToolDescriptor


class MultiToolPlanningErrorCode(str, Enum):
    """Stable result codes for deterministic multi-tool planning."""

    MULTI_TOOL_PATTERN_NOT_FOUND = "MULTI_TOOL_PATTERN_NOT_FOUND"
    MULTI_TOOL_PATTERN_AMBIGUOUS = "MULTI_TOOL_PATTERN_AMBIGUOUS"
    MISSING_SOURCE = "MISSING_SOURCE"
    MISSING_DESTINATION = "MISSING_DESTINATION"
    REQUIRED_TOOL_UNAVAILABLE = "REQUIRED_TOOL_UNAVAILABLE"
    OUTPUT_CONTRACT_UNKNOWN = "OUTPUT_CONTRACT_UNKNOWN"
    INVALID_MULTI_TOOL_PLAN = "INVALID_MULTI_TOOL_PLAN"
    MULTI_TOOL_PLANNING_FAILED = "MULTI_TOOL_PLANNING_FAILED"


@dataclass(frozen=True, slots=True)
class MultiToolPlanningResult:
    """Structured result for deterministic multi-tool planning."""

    success: bool
    handled: bool
    objective: str
    plan: ExecutionPlan | None = None
    matched_pattern: str | None = None
    selected_tools: tuple[str, ...] = ()
    requires_clarification: bool = False
    missing_information: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence: float = 0.0
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class MultiToolPattern:
    """One explicit deterministic multi-tool pattern."""

    name: str
    required_capabilities: tuple[str, ...]
    required_information: tuple[str, ...]
    risk_implications: tuple[str, ...]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    priority: int
    matcher: Callable[[str], bool]
    builder: Callable[
        ["DeterministicMultiToolPlanner", str, SemanticToolCatalog, ToolSelector, "MultiToolPattern"],
        MultiToolPlanningResult,
    ]


@dataclass(frozen=True, slots=True)
class _ExtractedPaths:
    paths: tuple[str, ...]
    directory: str | None = None
    filename: str | None = None


class DeterministicMultiToolPlanner:
    """Build valid ExecutionPlan objects for a small set of safe patterns."""

    def __init__(
        self,
        validator: ExecutionPlanValidator | None = None,
    ) -> None:
        self._validator = validator or ExecutionPlanValidator()
        self._patterns = tuple(
            sorted(
                (
                    MultiToolPattern(
                        name="read_then_write",
                        required_capabilities=("read_file", "write_file"),
                        required_information=("source_path", "destination_path"),
                        risk_implications=("write_file may create or overwrite local files",),
                        positive_examples=(
                            "lee origen.txt y guarda el contenido en copia.txt",
                            "copia el contenido de A en B",
                        ),
                        negative_examples=("lee un archivo y explicalo",),
                        priority=10,
                        matcher=_matches_read_then_write,
                        builder=DeterministicMultiToolPlanner._build_read_then_write,
                    ),
                    MultiToolPattern(
                        name="list_then_read",
                        required_capabilities=("list_directory", "read_file"),
                        required_information=("directory_path", "file_name"),
                        risk_implications=(),
                        positive_examples=(
                            "lista los archivos de C:/Temp y lee notas.txt",
                        ),
                        negative_examples=("lista archivos y resume su contenido",),
                        priority=20,
                        matcher=_matches_list_then_read,
                        builder=DeterministicMultiToolPlanner._build_list_then_read,
                    ),
                    MultiToolPattern(
                        name="process_find_then_terminate",
                        required_capabilities=("find_process", "terminate_process"),
                        required_information=("process_query",),
                        risk_implications=("terminate_process can stop a running process",),
                        positive_examples=("encuentra notepad y terminalo",),
                        negative_examples=("explica que es un proceso",),
                        priority=30,
                        matcher=_matches_process_find_then_terminate,
                        builder=DeterministicMultiToolPlanner._build_process_find_then_terminate,
                    ),
                ),
                key=lambda pattern: pattern.priority,
            )
        )

    def plan(
        self,
        objective: str,
        catalog: SemanticToolCatalog,
        selector: ToolSelector,
    ) -> MultiToolPlanningResult:
        """Return a deterministic multi-tool plan without executing anything."""
        normalized = _normalize(objective)
        if not normalized:
            return MultiToolPlanningResult(
                success=False,
                handled=False,
                objective=objective,
                errors=("Objective is empty.",),
                error_code=MultiToolPlanningErrorCode.MULTI_TOOL_PATTERN_NOT_FOUND.value,
            )

        matched = tuple(
            pattern
            for pattern in self._patterns
            if pattern.matcher(normalized) and not _matches_negative_example(normalized, pattern)
        )
        if not matched:
            return MultiToolPlanningResult(
                success=False,
                handled=False,
                objective=objective,
                error_code=MultiToolPlanningErrorCode.MULTI_TOOL_PATTERN_NOT_FOUND.value,
            )

        if len(matched) > 1:
            return MultiToolPlanningResult(
                success=False,
                handled=True,
                objective=objective,
                matched_pattern=None,
                requires_clarification=True,
                errors=("Multiple deterministic multi-tool patterns matched the objective.",),
                confidence=0.35,
                error_code=MultiToolPlanningErrorCode.MULTI_TOOL_PATTERN_AMBIGUOUS.value,
            )

        pattern = matched[0]
        return pattern.builder(self, objective, catalog, selector, pattern)

    def patterns(self) -> tuple[MultiToolPattern, ...]:
        """Return supported deterministic patterns."""
        return self._patterns

    def _build_read_then_write(
        self,
        objective: str,
        catalog: SemanticToolCatalog,
        selector: ToolSelector,
        pattern: MultiToolPattern,
    ) -> MultiToolPlanningResult:
        availability = _capability_tools(catalog, pattern.required_capabilities)
        if availability.missing:
            return _missing_tools_result(objective, pattern, availability)

        selector_check = _selector_check(objective, catalog, selector, availability.tools)
        if selector_check.errors:
            return _selector_blocked_result(objective, pattern, availability.tools, selector_check)

        paths = _extract_paths(objective)
        missing: list[str] = []
        destination_only = len(paths.paths) == 1 and _looks_destination_context(objective, paths.paths[0])
        if len(paths.paths) < 1 or destination_only:
            missing.append("source_path")
        if len(paths.paths) < 2 and not destination_only:
            missing.append("destination_path")
        if missing:
            return _missing_information_result(objective, pattern, availability.tools, missing)

        assert len(paths.paths) >= 2
        steps = [
            ExecutionStep(
                id="step_1",
                description="Read source file content.",
                tool=availability.tools[0],
                arguments={"path": paths.paths[0]},
            ),
            ExecutionStep(
                id="step_2",
                description="Write source content to destination file.",
                tool=availability.tools[1],
                dependencies=("step_1",),
                arguments={
                    "path": paths.paths[1],
                    "content": StepOutputReference("step_1"),
                },
            ),
        ]
        verification_requested = _requests_written_content_verification(objective)
        if verification_requested:
            steps.append(
                ExecutionStep(
                    id="step_3",
                    description="Read destination file to verify written content.",
                    tool=availability.tools[0],
                    dependencies=("step_2",),
                    arguments={"path": paths.paths[1]},
                )
            )

        plan = ExecutionPlan(
            goal=objective.strip(),
            ordered_steps=tuple(steps),
            estimated_steps=len(steps),
            required_tools=availability.tools,
            detected_risks=_plan_risks(catalog, availability.tools, pattern),
            requires_confirmation=_requires_confirmation(catalog, availability.tools),
            status="planned",
            acceptance_criteria=(
                _read_write_acceptance_criteria(paths.paths[1])
                if verification_requested
                else ()
            ),
        )
        return self._validated_result(objective, pattern, availability.tools, plan, selector_check.warnings, 0.92)

    def _build_list_then_read(
        self,
        objective: str,
        catalog: SemanticToolCatalog,
        selector: ToolSelector,
        pattern: MultiToolPattern,
    ) -> MultiToolPlanningResult:
        availability = _capability_tools(catalog, pattern.required_capabilities)
        if availability.missing:
            return _missing_tools_result(objective, pattern, availability)

        selector_check = _selector_check(objective, catalog, selector, availability.tools)
        if selector_check.errors:
            return _selector_blocked_result(objective, pattern, availability.tools, selector_check)

        paths = _extract_paths(objective)
        directory = paths.directory or (paths.paths[0] if paths.paths and not _looks_like_file_path(paths.paths[0]) else None)
        filename = paths.filename or next(
            (path for path in paths.paths if _looks_like_file_path(path)),
            None,
        )
        missing: list[str] = []
        if directory is None:
            missing.append("directory_path")
        if filename is None:
            missing.append("file_name")
        if missing:
            return _missing_information_result(objective, pattern, availability.tools, missing)

        assert directory is not None
        assert filename is not None
        read_path = filename if _looks_like_absolute_path(filename) else _join_path(directory, filename)
        plan = ExecutionPlan(
            goal=objective.strip(),
            ordered_steps=(
                ExecutionStep(
                    id="step_1",
                    description="List directory contents.",
                    tool=availability.tools[0],
                    arguments={"path": directory},
                ),
                ExecutionStep(
                    id="step_2",
                    description="Read requested file after listing directory.",
                    tool=availability.tools[1],
                    dependencies=("step_1",),
                    arguments={"path": read_path},
                ),
            ),
            estimated_steps=2,
            required_tools=availability.tools,
            detected_risks=_plan_risks(catalog, availability.tools, pattern),
            requires_confirmation=_requires_confirmation(catalog, availability.tools),
            status="planned",
        )
        return self._validated_result(objective, pattern, availability.tools, plan, selector_check.warnings, 0.88)

    def _build_process_find_then_terminate(
        self,
        objective: str,
        catalog: SemanticToolCatalog,
        selector: ToolSelector,
        pattern: MultiToolPattern,
    ) -> MultiToolPlanningResult:
        availability = _capability_tools(catalog, pattern.required_capabilities)
        if availability.missing:
            return _missing_tools_result(objective, pattern, availability)

        selector_check = _selector_check(objective, catalog, selector, availability.tools)
        if selector_check.errors:
            return _selector_blocked_result(objective, pattern, availability.tools, selector_check)

        process_query = _extract_process_query(objective)
        if process_query is None:
            return _missing_information_result(objective, pattern, availability.tools, ["process_query"])

        find_descriptor = catalog.get(availability.tools[0])
        terminate_descriptor = catalog.get(availability.tools[1])
        find_arg = _first_supported_argument(find_descriptor, ("query", "name", "process"))
        terminate_arg = _first_supported_argument(terminate_descriptor, ("pid", "query", "process"))
        if find_arg is None or terminate_arg is None:
            return MultiToolPlanningResult(
                success=False,
                handled=True,
                objective=objective,
                matched_pattern=pattern.name,
                selected_tools=availability.tools,
                requires_clarification=True,
                errors=("Process tools do not expose compatible technical arguments.",),
                confidence=0.55,
                error_code=MultiToolPlanningErrorCode.OUTPUT_CONTRACT_UNKNOWN.value,
            )
        if terminate_arg == "pid" and "pid" not in find_descriptor.output_fields:
            return MultiToolPlanningResult(
                success=False,
                handled=True,
                objective=objective,
                matched_pattern=pattern.name,
                selected_tools=availability.tools,
                requires_clarification=True,
                errors=("Find-process output contract does not declare field 'pid'.",),
                confidence=0.55,
                error_code=MultiToolPlanningErrorCode.OUTPUT_CONTRACT_UNKNOWN.value,
            )

        terminate_value: object = {"$ref": "steps.step_1.output.pid"} if terminate_arg == "pid" else {"$ref": "steps.step_1.output"}
        plan = ExecutionPlan(
            goal=objective.strip(),
            ordered_steps=(
                ExecutionStep(
                    id="step_1",
                    description="Find matching process.",
                    tool=availability.tools[0],
                    arguments={find_arg: process_query},
                ),
                ExecutionStep(
                    id="step_2",
                    description="Terminate matching process.",
                    tool=availability.tools[1],
                    dependencies=("step_1",),
                    arguments={terminate_arg: terminate_value},
                ),
            ),
            estimated_steps=2,
            required_tools=availability.tools,
            detected_risks=_plan_risks(catalog, availability.tools, pattern),
            requires_confirmation=True,
            status="planned",
        )
        return self._validated_result(objective, pattern, availability.tools, plan, selector_check.warnings, 0.82)

    def _validated_result(
        self,
        objective: str,
        pattern: MultiToolPattern,
        selected_tools: tuple[str, ...],
        plan: ExecutionPlan,
        warnings: tuple[str, ...],
        confidence: float,
    ) -> MultiToolPlanningResult:
        validation = self._validator.validate(plan)
        if not validation.is_valid:
            return MultiToolPlanningResult(
                success=False,
                handled=True,
                objective=objective,
                plan=plan,
                matched_pattern=pattern.name,
                selected_tools=selected_tools,
                requires_clarification=True,
                errors=tuple(validation.errors),
                warnings=warnings + tuple(validation.warnings),
                confidence=confidence,
                error_code=MultiToolPlanningErrorCode.INVALID_MULTI_TOOL_PLAN.value,
            )

        return MultiToolPlanningResult(
            success=True,
            handled=True,
            objective=objective,
            plan=plan,
            matched_pattern=pattern.name,
            selected_tools=selected_tools,
            warnings=warnings + tuple(validation.warnings),
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class _CapabilityAvailability:
    tools: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectorCheck:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _capability_tools(
    catalog: SemanticToolCatalog,
    capabilities: tuple[str, ...],
) -> _CapabilityAvailability:
    tools: list[str] = []
    missing: list[str] = []
    for capability in capabilities:
        matches = catalog.find_by_capability(capability)
        if not matches:
            missing.append(capability)
            continue
        tools.append(matches[0].name)
    return _CapabilityAvailability(tuple(tools), tuple(missing))


def _selector_check(
    objective: str,
    catalog: SemanticToolCatalog,
    selector: ToolSelector,
    required_tools: tuple[str, ...],
) -> _SelectorCheck:
    selection = selector.select_from_catalog(objective, catalog, top_k=5)
    if not selection.success:
        return _SelectorCheck(errors=tuple(selection.errors), warnings=tuple(selection.warnings))
    candidate_tools = {candidate.tool_name for candidate in selection.candidates}
    missing_candidates = tuple(tool for tool in required_tools if tool not in candidate_tools)
    if missing_candidates:
        return _SelectorCheck(
            errors=(
                "ToolSelector did not return required pattern tools: "
                + ", ".join(missing_candidates)
                + ".",
            ),
            warnings=tuple(selection.warnings),
        )
    return _SelectorCheck(warnings=tuple(selection.warnings))


def _selector_blocked_result(
    objective: str,
    pattern: MultiToolPattern,
    selected_tools: tuple[str, ...],
    selector_check: _SelectorCheck,
) -> MultiToolPlanningResult:
    return MultiToolPlanningResult(
        success=False,
        handled=True,
        objective=objective,
        matched_pattern=pattern.name,
        selected_tools=selected_tools,
        requires_clarification=True,
        errors=selector_check.errors,
        warnings=selector_check.warnings,
        confidence=0.45,
        error_code=MultiToolPlanningErrorCode.MULTI_TOOL_PLANNING_FAILED.value,
    )


def _missing_tools_result(
    objective: str,
    pattern: MultiToolPattern,
    availability: _CapabilityAvailability,
) -> MultiToolPlanningResult:
    return MultiToolPlanningResult(
        success=False,
        handled=True,
        objective=objective,
        matched_pattern=pattern.name,
        selected_tools=availability.tools,
        requires_clarification=True,
        missing_information=availability.missing,
        errors=("Required tools are unavailable: " + ", ".join(availability.missing) + ".",),
        confidence=0.5,
        error_code=MultiToolPlanningErrorCode.REQUIRED_TOOL_UNAVAILABLE.value,
    )


def _missing_information_result(
    objective: str,
    pattern: MultiToolPattern,
    selected_tools: tuple[str, ...],
    missing: list[str],
) -> MultiToolPlanningResult:
    error_code = (
        MultiToolPlanningErrorCode.MISSING_SOURCE.value
        if any(item in {"source_path", "directory_path", "file_name", "process_query"} for item in missing)
        else MultiToolPlanningErrorCode.MISSING_DESTINATION.value
    )
    if "destination_path" in missing:
        error_code = MultiToolPlanningErrorCode.MISSING_DESTINATION.value

    return MultiToolPlanningResult(
        success=False,
        handled=True,
        objective=objective,
        matched_pattern=pattern.name,
        selected_tools=selected_tools,
        requires_clarification=True,
        missing_information=tuple(missing),
        errors=("Missing required information: " + ", ".join(missing) + ".",),
        confidence=0.65,
        error_code=error_code,
    )


def _plan_risks(
    catalog: SemanticToolCatalog,
    tools: tuple[str, ...],
    pattern: MultiToolPattern,
) -> tuple[str, ...]:
    risks: list[str] = ["Multi-step deterministic plan must preserve dependency order."]
    risks.extend(pattern.risk_implications)
    for tool in tools:
        descriptor = catalog.get(tool)
        if descriptor.requires_confirmation:
            risks.append(f"Tool '{tool}' requires confirmation.")
        if descriptor.risk_level in {"high", "critical"}:
            risks.append(f"Tool '{tool}' has {descriptor.risk_level} risk.")
    return tuple(dict.fromkeys(risks))


def _requires_confirmation(
    catalog: SemanticToolCatalog,
    tools: tuple[str, ...],
) -> bool:
    return any(catalog.get(tool).requires_confirmation for tool in tools)


def _matches_read_then_write(normalized: str) -> bool:
    has_read = bool(re.search(r"\b(?:lee|leer|copia|copiar)\b", normalized))
    has_write = bool(re.search(r"\b(?:guarda|guardar|guardalo|guardala|escribe|copia|copiar)\b", normalized))
    has_content = (
        "contenido" in normalized
        or "copia" in normalized
        or "copiar" in normalized
        or "guardalo" in normalized
        or "guardala" in normalized
    )
    return has_read and has_write and has_content


def _requests_written_content_verification(objective: str) -> bool:
    normalized = _normalize(objective)
    return bool(
        re.search(
            r"\b(?:comprueba|comprobar|verifica|verificar|confirma|confirmar)\b",
            normalized,
        )
    )


def _read_write_acceptance_criteria(
    resource_path: str,
) -> tuple[AcceptanceCriterion, ...]:
    return (
        AcceptanceCriterion(
            "expected_step_count",
            AcceptanceCriterionKind.EXPECTED_STEP_COUNT,
            "The expected three-step chain completed.",
            expected_count=3,
        ),
        AcceptanceCriterion(
            "source_read_tool_used",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "The source was read with read_file.",
            source_step_id="step_1",
            tool_name="read_file",
        ),
        AcceptanceCriterion(
            "write_tool_used",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "The destination was written with write_file.",
            source_step_id="step_2",
            tool_name="write_file",
        ),
        AcceptanceCriterion(
            "verification_read_tool_used",
            AcceptanceCriterionKind.EXPECTED_TOOL_USED,
            "The produced resource was reopened with read_file.",
            source_step_id="step_3",
            tool_name="read_file",
        ),
        AcceptanceCriterion(
            "resource_exists",
            AcceptanceCriterionKind.RESOURCE_EXISTS,
            "The declared destination file exists.",
            source_step_id="step_2",
            resource_path=resource_path,
        ),
        AcceptanceCriterion(
            "resource_readable",
            AcceptanceCriterionKind.RESOURCE_READABLE,
            "The declared destination file is readable.",
            source_step_id="step_3",
            resource_path=resource_path,
        ),
        AcceptanceCriterion(
            "resource_content_equals_source",
            AcceptanceCriterionKind.RESOURCE_CONTENT_EQUALS,
            "The destination content equals the source output.",
            source_step_id="step_3",
            comparison_step_id="step_1",
            resource_path=resource_path,
        ),
        AcceptanceCriterion(
            "verification_output_equals_source",
            AcceptanceCriterionKind.OUTPUT_EQUALS,
            "The verification read equals the source read.",
            source_step_id="step_3",
            comparison_step_id="step_1",
        ),
        AcceptanceCriterion(
            "no_pending_confirmations",
            AcceptanceCriterionKind.NO_PENDING_CONFIRMATIONS,
            "No required confirmation remains pending.",
        ),
        AcceptanceCriterion(
            "no_critical_failures",
            AcceptanceCriterionKind.NO_CRITICAL_FAILURES,
            "No critical execution failure occurred.",
        ),
    )


def _matches_list_then_read(normalized: str) -> bool:
    has_list = bool(re.search(r"\b(?:lista|listar|busca|buscar)\b", normalized))
    has_read = bool(re.search(r"\b(?:lee|leer)\b", normalized))
    return has_list and has_read


def _matches_process_find_then_terminate(normalized: str) -> bool:
    has_find = bool(re.search(r"\b(?:encuentra|busca|buscar|localiza)\b", normalized))
    has_terminate = bool(re.search(r"\b(?:terminalo|termina|terminar|mata|matar)\b", normalized))
    return has_find and has_terminate and "proceso" in normalized


def _matches_negative_example(
    normalized: str,
    pattern: MultiToolPattern,
) -> bool:
    return any(_normalize(example) == normalized for example in pattern.negative_examples)


def _extract_paths(
    objective: str,
) -> _ExtractedPaths:
    paths = tuple(
        match.group("path").strip().rstrip(".,;")
        for match in re.finditer(
            r"(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w.-]+[\\/])*[\w.-]+\.[A-Za-z0-9]{1,8}|[A-Za-z]:[\\/][\w.-]+(?:[\\/][\w.-]+)*|(?:[\w.-]+[\\/])+[\w.-]+)",
            objective,
        )
    )
    directory_match = re.search(
        r"\b(?:en|de|desde|directorio|carpeta|ruta)\s+(?P<directory>(?:[A-Za-z]:[\\/])?[\w./\\-]+)",
        objective,
        flags=re.IGNORECASE,
    )
    filename_match = re.search(
        r"\b(?:lee|leer)\s+(?:el\s+archivo\s+)?(?P<filename>[\w.-]+\.[A-Za-z0-9]{1,8})\b",
        objective,
        flags=re.IGNORECASE,
    )
    directory = None
    if directory_match:
        candidate = directory_match.group("directory").strip().rstrip(".,;")
        if not _looks_like_file_path(candidate):
            directory = candidate
    filename = filename_match.group("filename").strip() if filename_match else None
    return _ExtractedPaths(paths=paths, directory=directory, filename=filename)


def _extract_process_query(
    objective: str,
) -> str | None:
    match = re.search(
        r"\b(?:proceso\s+)?(?P<query>[\w.-]+)\s+(?:y\s+)?(?:terminalo|termina|terminar|mata|matar)\b",
        objective,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group("query").strip()
        if value.lower() not in {"proceso", "el", "lo"}:
            return value

    match = re.search(
        r"\b(?:encuentra|busca|localiza)\s+(?P<query>[\w.-]+)\b",
        objective,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group("query").strip()
        if value.lower() not in {"proceso", "el", "lo"}:
            return value
    return None


def _first_supported_argument(
    descriptor: SemanticToolDescriptor,
    names: tuple[str, ...],
) -> str | None:
    supported = set(descriptor.technical_arguments or descriptor.required_arguments + descriptor.optional_arguments)
    for name in names:
        if name in supported:
            return name
    return None


def _looks_like_file_path(
    value: str,
) -> bool:
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", value))


def _looks_like_absolute_path(
    value: str,
) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _looks_destination_context(
    objective: str,
    path: str,
) -> bool:
    return bool(
        re.search(
            r"\b(?:en|a|hacia)\s+" + re.escape(path) + r"\b",
            objective,
            flags=re.IGNORECASE,
        )
    )


def _join_path(
    directory: str,
    filename: str,
) -> str:
    separator = "\\" if "\\" in directory else "/"
    return directory.rstrip("/\\") + separator + filename


def _normalize(
    text: str,
) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s./\\:-]", " ", without_accents)
    return " ".join(without_punctuation.split())
