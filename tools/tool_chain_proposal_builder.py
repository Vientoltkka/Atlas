"""Build structured tool-chain proposals from natural-language requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata

from tools.argument_schema import ArgumentValidationError, ArgumentValidator
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.intent_selector import ToolIntent, ToolSelector
from tools.tool_chain_runner import ToolChainStep
from tools.tool_proposal_builder import (
    StructuredToolProposal,
    ToolProposalBuilder,
    ToolProposalStatus,
)


_REFERENCE_PATTERN = re.compile(
    r"\$\{steps\.([A-Za-z0-9_-]+)\.(output|result)(?:\.([A-Za-z0-9_.-]+))?\}"
)


class ToolChainProposalStatus(str, Enum):
    """Lifecycle states for a proposed linear tool chain."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


class ToolChainProposalError(ValueError):
    """Raised when a chain proposal cannot become a chain definition."""


@dataclass(frozen=True, slots=True)
class StructuredToolChainStepProposal:
    """One immutable proposed step in a linear tool chain."""

    id: str
    tool_name: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    status: ToolProposalStatus = ToolProposalStatus.UNSUPPORTED
    missing_arguments: tuple[str, ...] = ()
    ambiguous_arguments: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(
                {
                    name: _freeze_value(value)
                    for name, value in self.arguments.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class StructuredToolChainProposal:
    """Structured, validated proposal before any chain execution."""

    steps: tuple[StructuredToolChainStepProposal, ...] = ()
    status: ToolChainProposalStatus = ToolChainProposalStatus.UNSUPPORTED
    confidence: float = 0.0
    reason: str = ""
    missing_information: tuple[str, ...] = ()
    ambiguous_information: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()
    source_text: str = ""

    def __post_init__(self) -> None:
        confidence = min(max(float(self.confidence), 0.0), 1.0)
        object.__setattr__(self, "confidence", confidence)

    @property
    def executable(self) -> bool:
        """Return whether this proposal can become ToolChainStep objects."""
        return self.status is ToolChainProposalStatus.COMPLETE

    def to_tool_chain_definition(
        self,
        selector: ToolSelector,
        validator: ArgumentValidator,
    ) -> tuple[ToolChainStep, ...]:
        """Revalidate and return the exact definition consumed by ToolChainRunner."""
        if not self.executable:
            raise ToolChainProposalError(
                "Only COMPLETE tool-chain proposals can be converted."
            )

        _validate_step_ids(self.steps)
        _validate_references(self.steps)

        chain_steps: list[ToolChainStep] = []
        for step in self.steps:
            if step.tool_name is None:
                raise ToolChainProposalError("Complete steps require tool_name.")

            arguments = _thaw_arguments(step.arguments)
            validator.validate(selector.select(ToolIntent(step.tool_name, arguments)))
            chain_steps.append(
                ToolChainStep(
                    step_id=step.id,
                    tool_name=step.tool_name,
                    arguments=arguments,
                )
            )

        return tuple(chain_steps)


class ToolChainProposalBuilder:
    """Convert a TOOL_CHAIN decision into a safe linear chain proposal."""

    def __init__(
        self,
        proposal_builder: ToolProposalBuilder,
        selector: ToolSelector,
        validator: ArgumentValidator,
    ) -> None:
        self._proposal_builder = proposal_builder
        self._selector = selector
        self._validator = validator

    def build(
        self,
        source_text: str,
        decision: ExecutionDecision,
        candidate_tools: tuple[str, ...] | None = None,
    ) -> StructuredToolChainProposal:
        """Build one linear chain proposal without executing any tool."""
        candidates = tuple(candidate_tools or decision.candidate_tools)

        if decision.mode is not ExecutionMode.TOOL_CHAIN:
            return self._unsupported(
                (),
                source_text,
                f"Execution decision mode {decision.mode.value} is not TOOL_CHAIN.",
                decision.confidence,
            )

        if len(candidates) < 2:
            return self._unsupported(
                (),
                source_text,
                "TOOL_CHAIN proposals require at least two candidate tools.",
                decision.confidence,
            )

        segments = _split_segments(source_text)
        steps: list[StructuredToolChainStepProposal] = []

        for index, tool_name in enumerate(candidates):
            segment = _segment_for_tool(tool_name, segments, index, source_text)
            single = self._build_single_step(segment, decision, tool_name)
            step_id = _unique_step_id(_base_step_id(tool_name), steps)
            previous_steps = tuple(steps)
            step = self._to_chain_step(step_id, single, segment, previous_steps)
            steps.append(step)

        validation_errors = _reference_validation_errors(tuple(steps))
        missing = _missing_information(tuple(steps))
        ambiguous = _ambiguous_information(tuple(steps))
        status = _chain_status(tuple(steps), validation_errors)

        return StructuredToolChainProposal(
            steps=tuple(steps),
            status=status,
            confidence=_chain_confidence(decision.confidence, status),
            reason=_chain_reason(decision.reason, status),
            missing_information=missing,
            ambiguous_information=ambiguous,
            validation_errors=validation_errors,
            source_text=source_text,
        )

    def to_tool_chain_definition(
        self,
        proposal: StructuredToolChainProposal,
    ) -> tuple[ToolChainStep, ...]:
        """Revalidate and return ToolChainRunner input without executing it."""
        return proposal.to_tool_chain_definition(self._selector, self._validator)

    def _build_single_step(
        self,
        segment: str,
        decision: ExecutionDecision,
        tool_name: str,
    ) -> StructuredToolProposal:
        single_decision = ExecutionDecision(
            mode=ExecutionMode.SINGLE_TOOL,
            reason=decision.reason,
            confidence=decision.confidence,
            candidate_tools=(tool_name,),
            required_capabilities=decision.required_capabilities,
        )
        return self._proposal_builder.build(segment, single_decision)

    def _to_chain_step(
        self,
        step_id: str,
        proposal: StructuredToolProposal,
        segment: str,
        previous_steps: tuple[StructuredToolChainStepProposal, ...],
    ) -> StructuredToolChainStepProposal:
        arguments = dict(proposal.arguments)
        depends_on: tuple[str, ...] = ()

        if proposal.tool_name == "file.write":
            reference = _reference_for_write(segment, previous_steps)
            if reference is not None:
                arguments["content"] = reference
                depends_on = _reference_step_ids(reference)

        references = _references_in_arguments(arguments)
        missing_arguments = tuple(
            item
            for item in proposal.missing_arguments
            if item not in arguments
        )

        try:
            validation_errors = _validation_errors(
                proposal.tool_name,
                arguments,
                self._selector,
                self._validator,
            )
        except ToolChainProposalError as error:
            validation_errors = (str(error),)

        proposal_validation_errors = _remaining_validation_errors(
            proposal.validation_errors,
            arguments,
        )
        status = _step_status(proposal, missing_arguments, validation_errors)

        return StructuredToolChainStepProposal(
            id=step_id,
            tool_name=proposal.tool_name,
            arguments=arguments,
            depends_on=depends_on,
            references=references,
            status=status,
            missing_arguments=missing_arguments,
            ambiguous_arguments=proposal.ambiguous_arguments,
            validation_errors=_merge_unique(
                proposal_validation_errors + validation_errors
            ),
            reason=proposal.reason,
        )

    def _unsupported(
        self,
        steps: tuple[StructuredToolChainStepProposal, ...],
        source_text: str,
        reason: str,
        confidence: float,
    ) -> StructuredToolChainProposal:
        return StructuredToolChainProposal(
            steps=steps,
            status=ToolChainProposalStatus.UNSUPPORTED,
            confidence=min(confidence, 0.35),
            reason=reason,
            source_text=source_text,
        )


def _split_segments(source_text: str) -> tuple[str, ...]:
    parts = re.split(
        r"\s*(?:,|\by\b|\bdespues\b|\bdespu.s\b|\bluego\b|\bentonces\b)\s*",
        source_text,
        flags=re.IGNORECASE,
    )
    return tuple(part.strip() for part in parts if part.strip())


def _segment_for_tool(
    tool_name: str,
    segments: tuple[str, ...],
    index: int,
    fallback: str,
) -> str:
    for segment in segments:
        normalized = _normalize(segment)
        if _segment_matches_tool(tool_name, normalized):
            return segment

    if index < len(segments):
        return segments[index]

    return fallback


def _segment_matches_tool(tool_name: str, normalized: str) -> bool:
    patterns = {
        "file.read": (r"\blee", r"\bmuestra"),
        "directory.list": (r"\blista", r"\blistar"),
        "file.write": (r"\bescribe", r"\bcopia", r"\bguarda", r"\bcrea"),
        "desktop.application.open": (r"\babre", r"\babrir"),
        "desktop.text.type": (r"\bescribe", r"\bteclea"),
        "desktop.hotkey.press": (r"\bpulsa", r"\batajo"),
    }
    return any(
        re.search(pattern, normalized)
        for pattern in patterns.get(tool_name, ())
    )


def _base_step_id(tool_name: str) -> str:
    names = {
        "file.read": "read",
        "file.write": "write",
        "directory.list": "list",
        "desktop.application.open": "open",
        "desktop.text.type": "type",
        "desktop.hotkey.press": "hotkey",
    }
    return names.get(tool_name, tool_name.replace(".", "_"))


def _unique_step_id(
    base: str,
    previous_steps: list[StructuredToolChainStepProposal],
) -> str:
    used = {step.id for step in previous_steps}
    if base not in used:
        return base

    index = 2
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def _reference_for_write(
    segment: str,
    previous_steps: tuple[StructuredToolChainStepProposal, ...],
) -> str | None:
    if not previous_steps:
        return None

    normalized = _normalize(segment)
    if not re.search(r"\b(?:contenido|resultado|salida|ambos|dos veces|guardalo|guardala)\b", normalized):
        return None

    if "ambos" in normalized and len(previous_steps) >= 2:
        refs = [_reference_for_step(step) for step in previous_steps[-2:]]
        return "\n".join(ref for ref in refs if ref is not None) or None

    if "dos veces" in normalized:
        reference = _reference_for_step(previous_steps[-1])
        if reference is None:
            return None
        return f"{reference}\n{reference}"

    return _reference_for_step(previous_steps[-1])


def _reference_for_step(step: StructuredToolChainStepProposal) -> str | None:
    if step.tool_name == "file.read":
        return f"${{steps.{step.id}.output.content}}"

    if step.tool_name == "directory.list":
        return f"${{steps.{step.id}.output}}"

    return f"${{steps.{step.id}.output}}"


def _reference_step_ids(reference: str) -> tuple[str, ...]:
    ids: list[str] = []
    for match in _REFERENCE_PATTERN.finditer(reference):
        step_id = match.group(1)
        if step_id not in ids:
            ids.append(step_id)
    return tuple(ids)


def _references_in_arguments(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    for value in arguments.values():
        refs.extend(_references_in_value(value))
    return tuple(refs)


def _references_in_value(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(match.group(0) for match in _REFERENCE_PATTERN.finditer(value))

    if isinstance(value, tuple | list):
        refs: list[str] = []
        for item in value:
            refs.extend(_references_in_value(item))
        return tuple(refs)

    if isinstance(value, Mapping):
        refs: list[str] = []
        for item in value.values():
            refs.extend(_references_in_value(item))
        return tuple(refs)

    return ()


def _validation_errors(
    tool_name: str | None,
    arguments: Mapping[str, Any],
    selector: ToolSelector,
    validator: ArgumentValidator,
) -> tuple[str, ...]:
    if tool_name is None:
        return ("tool_name is missing",)

    try:
        validator.validate(selector.select(ToolIntent(tool_name, _thaw_arguments(arguments))))
    except ArgumentValidationError as error:
        return (f"{error.field}: {error.reason}",)
    except Exception as error:
        raise ToolChainProposalError(str(error)) from error

    return ()


def _remaining_validation_errors(
    validation_errors: tuple[str, ...],
    arguments: Mapping[str, Any],
) -> tuple[str, ...]:
    remaining: list[str] = []
    for error in validation_errors:
        field = error.split(":", 1)[0]
        if field in arguments and "required argument is missing" in error:
            continue
        remaining.append(error)
    return tuple(remaining)


def _merge_unique(items: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for item in items:
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def _step_status(
    proposal: StructuredToolProposal,
    missing_arguments: tuple[str, ...],
    validation_errors: tuple[str, ...],
) -> ToolProposalStatus:
    if proposal.status is ToolProposalStatus.UNSUPPORTED:
        return ToolProposalStatus.UNSUPPORTED

    if proposal.ambiguous_arguments:
        return ToolProposalStatus.AMBIGUOUS

    if missing_arguments or any("required argument is missing" in item for item in validation_errors):
        return ToolProposalStatus.INCOMPLETE

    if validation_errors:
        return ToolProposalStatus.UNSUPPORTED

    return ToolProposalStatus.COMPLETE


def _missing_information(
    steps: tuple[StructuredToolChainStepProposal, ...],
) -> tuple[str, ...]:
    missing: list[str] = []
    for step in steps:
        for argument in step.missing_arguments:
            missing.append(f"{step.id}.{argument}")
    return tuple(missing)


def _ambiguous_information(
    steps: tuple[StructuredToolChainStepProposal, ...],
) -> tuple[str, ...]:
    ambiguous: list[str] = []
    for step in steps:
        for argument in step.ambiguous_arguments:
            ambiguous.append(f"{step.id}.{argument}")
    return tuple(ambiguous)


def _chain_status(
    steps: tuple[StructuredToolChainStepProposal, ...],
    validation_errors: tuple[str, ...],
) -> ToolChainProposalStatus:
    if validation_errors:
        return ToolChainProposalStatus.UNSUPPORTED

    statuses = tuple(step.status for step in steps)
    if any(status is ToolProposalStatus.UNSUPPORTED for status in statuses):
        return ToolChainProposalStatus.UNSUPPORTED

    if any(status is ToolProposalStatus.AMBIGUOUS for status in statuses):
        return ToolChainProposalStatus.AMBIGUOUS

    if any(status is ToolProposalStatus.INCOMPLETE for status in statuses):
        return ToolChainProposalStatus.INCOMPLETE

    return ToolChainProposalStatus.COMPLETE


def _chain_confidence(
    confidence: float,
    status: ToolChainProposalStatus,
) -> float:
    if status is ToolChainProposalStatus.COMPLETE:
        return confidence
    if status is ToolChainProposalStatus.INCOMPLETE:
        return min(confidence, 0.65)
    if status is ToolChainProposalStatus.AMBIGUOUS:
        return min(confidence, 0.55)
    return min(confidence, 0.35)


def _chain_reason(
    decision_reason: str,
    status: ToolChainProposalStatus,
) -> str:
    return f"{decision_reason} Built linear chain proposal with status {status.value}."


def _reference_validation_errors(
    steps: tuple[StructuredToolChainStepProposal, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    seen: set[str] = set()

    for index, step in enumerate(steps):
        if step.id in seen:
            errors.append(f"duplicate step id: {step.id}")
        seen.add(step.id)

        for reference in step.references:
            match = _REFERENCE_PATTERN.fullmatch(reference)
            if match is None:
                errors.append(f"{step.id}: invalid reference syntax: {reference}")
                continue

            referenced_id = match.group(1)
            path = match.group(3)
            previous_ids = {previous.id for previous in steps[:index]}
            all_ids = {item.id for item in steps}

            if referenced_id not in all_ids:
                errors.append(f"{step.id}: reference step does not exist: {referenced_id}")
                continue

            if referenced_id not in previous_ids:
                errors.append(f"{step.id}: reference must point to a previous step: {referenced_id}")
                continue

            source = next(item for item in steps if item.id == referenced_id)
            if not _reference_path_supported(source, path):
                errors.append(f"{step.id}: reference field is not supported: {referenced_id}.{path}")

    return tuple(errors)


def _validate_step_ids(steps: tuple[StructuredToolChainStepProposal, ...]) -> None:
    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise ToolChainProposalError("duplicate step ids are not allowed")

    if any(not item for item in ids):
        raise ToolChainProposalError("step ids cannot be empty")


def _validate_references(steps: tuple[StructuredToolChainStepProposal, ...]) -> None:
    errors = _reference_validation_errors(steps)
    if errors:
        raise ToolChainProposalError(errors[0])


def _reference_path_supported(
    source: StructuredToolChainStepProposal,
    path: str | None,
) -> bool:
    if path is None:
        return True

    if source.tool_name == "file.read":
        return path == "content"

    if source.tool_name == "directory.list":
        return path.isdigit()

    return False


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                nested_name: _freeze_value(nested_value)
                for nested_name, nested_value in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)

    return value


def _thaw_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _thaw_value(value)
        for name, value in arguments.items()
    }


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            nested_name: _thaw_value(nested_value)
            for nested_name, nested_value in value.items()
        }

    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]

    return value


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_accents.split())
