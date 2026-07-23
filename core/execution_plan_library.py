"""Structured catalog for reusable Atlas execution plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

from core.execution_plan_registry import (
    ExecutionPlanAlreadyRegisteredError,
    ExecutionPlanReference,
    ExecutionPlanRegistry,
    ExecutionPlanRegistryError,
    RegisteredExecutionPlan,
    validate_plan_id,
    validate_plan_version,
)
from core.execution_plan_validator import plan_signature
from core.planner import ExecutionPlan, ExecutionStep


MAX_LIBRARY_WORKFLOWS = 256
MAX_WORKFLOW_TAGS = 32
MAX_WORKFLOW_TITLE_LENGTH = 120
MAX_WORKFLOW_DESCRIPTION_LENGTH = 2000
MAX_LIBRARY_ID_LENGTH = 128
MAX_CATEGORY_LENGTH = 128
MAX_TAG_LENGTH = 64

_CATEGORY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class ExecutionPlanLibraryError(RuntimeError):
    """Base error for execution-plan library operations."""

    def __init__(
        self,
        message: str,
        *,
        library_id: str | None = None,
        library_version: str | None = None,
        reference: ExecutionPlanReference | None = None,
        operation: str,
        code: str,
        reason: str | None = None,
        atomic: bool | None = None,
        rollback_performed: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.library_id = library_id
        self.library_version = library_version
        self.reference = reference
        self.operation = operation
        self.code = code
        self.reason = reason or message
        self.atomic = atomic
        self.rollback_performed = rollback_performed


class InvalidExecutionPlanLibraryError(ExecutionPlanLibraryError):
    """Raised when a library is malformed."""


class InvalidExecutionPlanLibraryIdError(InvalidExecutionPlanLibraryError):
    """Raised when a library id is invalid."""


class InvalidExecutionPlanLibraryVersionError(InvalidExecutionPlanLibraryError):
    """Raised when a library version is invalid."""


class InvalidWorkflowDefinitionError(InvalidExecutionPlanLibraryError):
    """Raised when a workflow definition is malformed."""


class InvalidWorkflowTitleError(InvalidWorkflowDefinitionError):
    """Raised when a workflow title is invalid."""


class InvalidWorkflowDescriptionError(InvalidWorkflowDefinitionError):
    """Raised when a workflow description is invalid."""


class InvalidWorkflowCategoryError(InvalidWorkflowDefinitionError):
    """Raised when a workflow category is invalid."""


class InvalidWorkflowTagError(InvalidWorkflowDefinitionError):
    """Raised when a workflow tag is invalid."""


class DuplicateWorkflowDefinitionError(InvalidExecutionPlanLibraryError):
    """Raised when a library contains duplicate workflow references."""


class ExecutionPlanLibraryTooLargeError(InvalidExecutionPlanLibraryError):
    """Raised when a library exceeds explicit size limits."""


class ExecutionPlanLibraryConflictError(ExecutionPlanLibraryError):
    """Raised when installing or uninstalling would affect a conflicting plan."""


class ExecutionPlanLibraryInstallError(ExecutionPlanLibraryError):
    """Raised when installing a library fails."""


class ExecutionPlanLibraryRollbackError(ExecutionPlanLibraryInstallError):
    """Raised when an atomic install rollback fails."""


class ExecutionPlanLibraryUninstallError(ExecutionPlanLibraryError):
    """Raised when uninstalling a library fails."""


class ExecutionPlanLibraryNotInstalledError(ExecutionPlanLibraryUninstallError):
    """Raised when strict uninstall expects an installed workflow."""


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Immutable functional definition for one reusable workflow."""

    reference: ExecutionPlanReference
    plan: ExecutionPlan
    title: str
    description: str
    category: str
    tags: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ExecutionPlanReference):
            raise InvalidWorkflowDefinitionError(
                "Workflow reference must be ExecutionPlanReference.",
                reference=None,
                operation="validate_workflow",
                code="INVALID_WORKFLOW_REFERENCE",
            )
        if not isinstance(self.plan, ExecutionPlan):
            raise InvalidWorkflowDefinitionError(
                "Workflow plan must be ExecutionPlan.",
                reference=self.reference,
                operation="validate_workflow",
                code="INVALID_WORKFLOW_PLAN",
            )
        _validate_plan_structure(self.plan, self.reference)
        object.__setattr__(
            self,
            "title",
            _validate_text(
                self.title,
                field_name="Workflow title",
                max_length=MAX_WORKFLOW_TITLE_LENGTH,
                error_type=InvalidWorkflowTitleError,
                reference=self.reference,
                operation="validate_workflow_title",
                code="INVALID_WORKFLOW_TITLE",
                allow_multiline=False,
            ),
        )
        object.__setattr__(
            self,
            "description",
            _validate_text(
                self.description,
                field_name="Workflow description",
                max_length=MAX_WORKFLOW_DESCRIPTION_LENGTH,
                error_type=InvalidWorkflowDescriptionError,
                reference=self.reference,
                operation="validate_workflow_description",
                code="INVALID_WORKFLOW_DESCRIPTION",
                allow_multiline=True,
            ),
        )
        object.__setattr__(
            self,
            "category",
            _validate_identifier(
                self.category,
                field_name="Workflow category",
                max_length=MAX_CATEGORY_LENGTH,
                pattern=_CATEGORY_PATTERN,
                error_type=InvalidWorkflowCategoryError,
                reference=self.reference,
                operation="validate_workflow_category",
                code="INVALID_WORKFLOW_CATEGORY",
            ),
        )
        object.__setattr__(self, "tags", _validate_tags(self.tags, self.reference))
        if not isinstance(self.enabled, bool):
            raise InvalidWorkflowDefinitionError(
                "Workflow enabled flag must be a bool.",
                reference=self.reference,
                operation="validate_workflow",
                code="INVALID_WORKFLOW_ENABLED",
            )


WorkflowDescriptor = WorkflowDefinition


@dataclass(frozen=True, slots=True)
class ExecutionPlanLibraryInstallResult:
    """Structured result for explicit library installation."""

    library_id: str
    library_version: str | None
    installed: tuple[ExecutionPlanReference, ...]
    replaced: tuple[ExecutionPlanReference, ...]
    skipped_disabled: tuple[ExecutionPlanReference, ...]
    atomic: bool
    failed_reference: ExecutionPlanReference | None = None
    rollback_performed: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionPlanLibraryUninstallResult:
    """Structured result for explicit library uninstallation."""

    library_id: str
    library_version: str | None
    removed: tuple[ExecutionPlanReference, ...]
    missing: tuple[ExecutionPlanReference, ...]
    conflicted: tuple[ExecutionPlanReference, ...]
    skipped_disabled: tuple[ExecutionPlanReference, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlanLibraryDefinition:
    """Immutable read model for a workflow library."""

    library_id: str
    version: str | None
    title: str | None
    description: str | None
    workflows: tuple[WorkflowDefinition, ...]


class ExecutionPlanLibrary:
    """Immutable catalog of reusable execution-plan workflows."""

    def __init__(
        self,
        library_id: str,
        workflows: Iterable[WorkflowDefinition],
        *,
        version: str | None = None,
        title: str | None = None,
        description: str | None = None,
        allow_empty: bool = False,
    ) -> None:
        self._library_id = _validate_library_id(library_id)
        self._version = _validate_library_version(version)
        self._title = (
            None
            if title is None
            else _validate_text(
                title,
                field_name="Library title",
                max_length=MAX_WORKFLOW_TITLE_LENGTH,
                error_type=InvalidExecutionPlanLibraryError,
                reference=None,
                operation="validate_library_title",
                code="INVALID_EXECUTION_PLAN_LIBRARY",
                allow_multiline=False,
            )
        )
        self._description = (
            None
            if description is None
            else _validate_text(
                description,
                field_name="Library description",
                max_length=MAX_WORKFLOW_DESCRIPTION_LENGTH,
                error_type=InvalidExecutionPlanLibraryError,
                reference=None,
                operation="validate_library_description",
                code="INVALID_EXECUTION_PLAN_LIBRARY",
                allow_multiline=True,
            )
        )
        workflow_tuple = tuple(workflows)
        if not workflow_tuple and not allow_empty:
            raise InvalidExecutionPlanLibraryError(
                "Execution plan library must contain at least one workflow.",
                library_id=self._library_id,
                library_version=self._version,
                operation="validate_library",
                code="EMPTY_EXECUTION_PLAN_LIBRARY",
            )
        if len(workflow_tuple) > MAX_LIBRARY_WORKFLOWS:
            raise ExecutionPlanLibraryTooLargeError(
                "Execution plan library exceeds the workflow limit.",
                library_id=self._library_id,
                library_version=self._version,
                operation="validate_library",
                code="EXECUTION_PLAN_LIBRARY_TOO_LARGE",
            )
        seen: set[ExecutionPlanReference] = set()
        for workflow in workflow_tuple:
            if not isinstance(workflow, WorkflowDefinition):
                raise InvalidWorkflowDefinitionError(
                    "Execution plan library workflows must be WorkflowDefinition.",
                    library_id=self._library_id,
                    library_version=self._version,
                    operation="validate_library",
                    code="INVALID_WORKFLOW_DEFINITION",
                )
            if workflow.reference in seen:
                raise DuplicateWorkflowDefinitionError(
                    "Execution plan library contains duplicate workflow references.",
                    library_id=self._library_id,
                    library_version=self._version,
                    reference=workflow.reference,
                    operation="validate_library",
                    code="DUPLICATE_WORKFLOW_DEFINITION",
                )
            seen.add(workflow.reference)
        self._workflows = workflow_tuple

    @property
    def library_id(self) -> str:
        return self._library_id

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def description(self) -> str | None:
        return self._description

    def definition(self) -> ExecutionPlanLibraryDefinition:
        return ExecutionPlanLibraryDefinition(
            library_id=self._library_id,
            version=self._version,
            title=self._title,
            description=self._description,
            workflows=self._workflows,
        )

    def workflows(self) -> tuple[WorkflowDefinition, ...]:
        return self._workflows

    def enabled_workflows(self) -> tuple[WorkflowDefinition, ...]:
        return tuple(workflow for workflow in self._workflows if workflow.enabled)

    def disabled_workflows(self) -> tuple[WorkflowDefinition, ...]:
        return tuple(workflow for workflow in self._workflows if not workflow.enabled)

    def get(self, reference: ExecutionPlanReference) -> WorkflowDefinition:
        for workflow in self._workflows:
            if workflow.reference == reference:
                return workflow
        raise ExecutionPlanLibraryError(
            "Workflow reference was not found in the library.",
            library_id=self._library_id,
            library_version=self._version,
            reference=reference,
            operation="get",
            code="WORKFLOW_NOT_FOUND",
        )

    def contains(self, reference: ExecutionPlanReference) -> bool:
        return any(workflow.reference == reference for workflow in self._workflows)

    def find_by_category(self, category: str) -> tuple[WorkflowDefinition, ...]:
        normalized = _validate_identifier(
            category,
            field_name="Workflow category",
            max_length=MAX_CATEGORY_LENGTH,
            pattern=_CATEGORY_PATTERN,
            error_type=InvalidWorkflowCategoryError,
            reference=None,
            operation="find_by_category",
            code="INVALID_WORKFLOW_CATEGORY",
        )
        return tuple(workflow for workflow in self._workflows if workflow.category == normalized)

    def find_by_tag(self, tag: str) -> tuple[WorkflowDefinition, ...]:
        normalized = _validate_identifier(
            tag,
            field_name="Workflow tag",
            max_length=MAX_TAG_LENGTH,
            pattern=_TAG_PATTERN,
            error_type=InvalidWorkflowTagError,
            reference=None,
            operation="find_by_tag",
            code="INVALID_WORKFLOW_TAG",
        )
        return tuple(workflow for workflow in self._workflows if normalized in workflow.tags)

    def search(
        self,
        *,
        category: str | None = None,
        tags: tuple[str, ...] = (),
        enabled: bool | None = None,
    ) -> tuple[WorkflowDefinition, ...]:
        if enabled is not None and not isinstance(enabled, bool):
            raise InvalidExecutionPlanLibraryError(
                "Search enabled filter must be bool or null.",
                library_id=self._library_id,
                library_version=self._version,
                operation="search",
                code="INVALID_SEARCH_FILTER",
            )
        normalized_category = (
            None
            if category is None
            else _validate_identifier(
                category,
                field_name="Workflow category",
                max_length=MAX_CATEGORY_LENGTH,
                pattern=_CATEGORY_PATTERN,
                error_type=InvalidWorkflowCategoryError,
                reference=None,
                operation="search",
                code="INVALID_WORKFLOW_CATEGORY",
            )
        )
        normalized_tags = _validate_tags(tags, None) if tags else ()
        return tuple(
            workflow
            for workflow in self._workflows
            if (normalized_category is None or workflow.category == normalized_category)
            and (enabled is None or workflow.enabled is enabled)
            and all(tag in workflow.tags for tag in normalized_tags)
        )

    def install(
        self,
        registry: ExecutionPlanRegistry,
        *,
        replace: bool = False,
        atomic: bool = True,
    ) -> ExecutionPlanLibraryInstallResult:
        if not isinstance(registry, ExecutionPlanRegistry):
            raise ExecutionPlanLibraryInstallError(
                "Library install requires an ExecutionPlanRegistry.",
                library_id=self._library_id,
                library_version=self._version,
                operation="install",
                code="INVALID_EXECUTION_PLAN_REGISTRY",
                atomic=atomic,
            )
        enabled = self.enabled_workflows()
        skipped_disabled = tuple(workflow.reference for workflow in self.disabled_workflows())
        originals = {
            workflow.reference: _registry_entry_for(registry, workflow.reference)
            for workflow in enabled
        }
        collisions = tuple(
            workflow.reference
            for workflow in enabled
            if originals[workflow.reference] is not None
        )
        if atomic and collisions and not replace:
            raise ExecutionPlanLibraryConflictError(
                "Execution plan library install found registered workflow collisions.",
                library_id=self._library_id,
                library_version=self._version,
                reference=collisions[0],
                operation="install",
                code="EXECUTION_PLAN_LIBRARY_CONFLICT",
                atomic=atomic,
                rollback_performed=False,
            )

        installed: list[ExecutionPlanReference] = []
        replaced: list[ExecutionPlanReference] = []
        touched: list[ExecutionPlanReference] = []
        try:
            for workflow in enabled:
                entry = RegisteredExecutionPlan(
                    reference=workflow.reference,
                    plan=workflow.plan,
                    description=workflow.description,
                    metadata={
                        "library_id": self._library_id,
                        "library_version": self._version,
                        "workflow_title": workflow.title,
                        "workflow_category": workflow.category,
                        "workflow_tags": workflow.tags,
                        "workflow_enabled": workflow.enabled,
                    },
                )
                registry.register_entry(entry, replace=replace)
                touched.append(workflow.reference)
                if originals[workflow.reference] is None:
                    installed.append(workflow.reference)
                else:
                    replaced.append(workflow.reference)
        except ExecutionPlanAlreadyRegisteredError:
            if not atomic:
                return ExecutionPlanLibraryInstallResult(
                    library_id=self._library_id,
                    library_version=self._version,
                    installed=tuple(installed),
                    replaced=tuple(replaced),
                    skipped_disabled=skipped_disabled,
                    atomic=atomic,
                    failed_reference=workflow.reference,
                    rollback_performed=False,
                )
            self._rollback_install(registry, touched, originals)
            raise ExecutionPlanLibraryConflictError(
                "Execution plan library install found registered workflow collisions.",
                library_id=self._library_id,
                library_version=self._version,
                reference=workflow.reference,
                operation="install",
                code="EXECUTION_PLAN_LIBRARY_CONFLICT",
                atomic=atomic,
                rollback_performed=True,
            )
        except Exception as error:
            if not atomic:
                return ExecutionPlanLibraryInstallResult(
                    library_id=self._library_id,
                    library_version=self._version,
                    installed=tuple(installed),
                    replaced=tuple(replaced),
                    skipped_disabled=skipped_disabled,
                    atomic=atomic,
                    failed_reference=workflow.reference,
                    rollback_performed=False,
                )
            try:
                self._rollback_install(registry, touched, originals)
            except Exception as rollback_error:
                raise ExecutionPlanLibraryRollbackError(
                    "Execution plan library install failed and rollback failed.",
                    library_id=self._library_id,
                    library_version=self._version,
                    reference=workflow.reference,
                    operation="install",
                    code="EXECUTION_PLAN_LIBRARY_ROLLBACK_FAILED",
                    reason=str(rollback_error),
                    atomic=atomic,
                    rollback_performed=False,
                ) from rollback_error
            raise ExecutionPlanLibraryInstallError(
                "Execution plan library install failed and was rolled back.",
                library_id=self._library_id,
                library_version=self._version,
                reference=workflow.reference,
                operation="install",
                code="EXECUTION_PLAN_LIBRARY_INSTALL_FAILED",
                reason=str(error),
                atomic=atomic,
                rollback_performed=True,
            ) from error

        return ExecutionPlanLibraryInstallResult(
            library_id=self._library_id,
            library_version=self._version,
            installed=tuple(installed),
            replaced=tuple(replaced),
            skipped_disabled=skipped_disabled,
            atomic=atomic,
        )

    def uninstall(
        self,
        registry: ExecutionPlanRegistry,
        *,
        strict: bool = False,
    ) -> ExecutionPlanLibraryUninstallResult:
        if not isinstance(registry, ExecutionPlanRegistry):
            raise ExecutionPlanLibraryUninstallError(
                "Library uninstall requires an ExecutionPlanRegistry.",
                library_id=self._library_id,
                library_version=self._version,
                operation="uninstall",
                code="INVALID_EXECUTION_PLAN_REGISTRY",
            )
        removed: list[ExecutionPlanReference] = []
        missing: list[ExecutionPlanReference] = []
        conflicted: list[ExecutionPlanReference] = []
        skipped_disabled = tuple(workflow.reference for workflow in self.disabled_workflows())

        for workflow in self.enabled_workflows():
            current_entry = _registry_entry_for(registry, workflow.reference)
            if current_entry is None:
                if strict:
                    raise ExecutionPlanLibraryNotInstalledError(
                        "Workflow is not installed in the registry.",
                        library_id=self._library_id,
                        library_version=self._version,
                        reference=workflow.reference,
                        operation="uninstall",
                        code="EXECUTION_PLAN_LIBRARY_NOT_INSTALLED",
                    )
                missing.append(workflow.reference)
                continue

            if plan_signature(current_entry.plan) != plan_signature(workflow.plan):
                if strict:
                    raise ExecutionPlanLibraryConflictError(
                        "Installed workflow plan signature differs from the library definition.",
                        library_id=self._library_id,
                        library_version=self._version,
                        reference=workflow.reference,
                        operation="uninstall",
                        code="EXECUTION_PLAN_LIBRARY_CONFLICT",
                    )
                conflicted.append(workflow.reference)
                continue

            registry.unregister(workflow.reference.plan_id, version=workflow.reference.version)
            removed.append(workflow.reference)

        return ExecutionPlanLibraryUninstallResult(
            library_id=self._library_id,
            library_version=self._version,
            removed=tuple(removed),
            missing=tuple(missing),
            conflicted=tuple(conflicted),
            skipped_disabled=skipped_disabled,
        )

    def library_signature(self) -> str:
        payload = {
            "library_id": self._library_id,
            "version": self._version,
            "workflows": [
                {
                    "reference": {
                        "plan_id": workflow.reference.plan_id,
                        "version": workflow.reference.version,
                    },
                    "title": workflow.title,
                    "category": workflow.category,
                    "tags": list(workflow.tags),
                    "enabled": workflow.enabled,
                    "plan_signature": plan_signature(workflow.plan),
                }
                for workflow in self._workflows
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _rollback_install(
        self,
        registry: ExecutionPlanRegistry,
        touched: list[ExecutionPlanReference],
        originals: dict[ExecutionPlanReference, RegisteredExecutionPlan | None],
    ) -> None:
        for reference in reversed(touched):
            original = originals[reference]
            if original is None:
                registry.unregister(reference.plan_id, version=reference.version)
            else:
                registry.register_entry(original, replace=True)


def _validate_library_id(value: str) -> str:
    try:
        validated = validate_plan_id(value)
    except ExecutionPlanRegistryError as error:
        raise InvalidExecutionPlanLibraryIdError(
            "Execution plan library id is invalid.",
            library_id=value if isinstance(value, str) else None,
            operation="validate_library_id",
            code="INVALID_EXECUTION_PLAN_LIBRARY_ID",
            reason=str(error),
        ) from error
    if len(validated) > MAX_LIBRARY_ID_LENGTH:
        raise InvalidExecutionPlanLibraryIdError(
            "Execution plan library id exceeds the length limit.",
            library_id=validated,
            operation="validate_library_id",
            code="INVALID_EXECUTION_PLAN_LIBRARY_ID",
        )
    return validated


def _validate_library_version(value: str | None) -> str | None:
    try:
        validated = validate_plan_version(value)
    except ExecutionPlanRegistryError as error:
        raise InvalidExecutionPlanLibraryVersionError(
            "Execution plan library version is invalid.",
            library_version=value if isinstance(value, str) else None,
            operation="validate_library_version",
            code="INVALID_EXECUTION_PLAN_LIBRARY_VERSION",
            reason=str(error),
        ) from error
    if validated is not None and validated.lower() == "latest":
        raise InvalidExecutionPlanLibraryVersionError(
            "Execution plan library version cannot be latest.",
            library_version=validated,
            operation="validate_library_version",
            code="INVALID_EXECUTION_PLAN_LIBRARY_VERSION",
        )
    return validated


def _validate_text(
    value: str,
    *,
    field_name: str,
    max_length: int,
    error_type: type[ExecutionPlanLibraryError],
    reference: ExecutionPlanReference | None,
    operation: str,
    code: str,
    allow_multiline: bool,
) -> str:
    if not isinstance(value, str):
        raise error_type(
            f"{field_name} must be a string.",
            reference=reference,
            operation=operation,
            code=code,
        )
    normalized = value.strip()
    if not normalized:
        raise error_type(
            f"{field_name} cannot be empty.",
            reference=reference,
            operation=operation,
            code=code,
        )
    if len(normalized) > max_length:
        raise error_type(
            f"{field_name} exceeds the length limit.",
            reference=reference,
            operation=operation,
            code=code,
        )
    for character in normalized:
        ordinal = ord(character)
        if ordinal < 32 and not (allow_multiline and character in "\r\n\t"):
            raise error_type(
                f"{field_name} cannot contain control characters.",
                reference=reference,
                operation=operation,
                code=code,
            )
    return normalized


def _validate_identifier(
    value: str,
    *,
    field_name: str,
    max_length: int,
    pattern: re.Pattern[str],
    error_type: type[ExecutionPlanLibraryError],
    reference: ExecutionPlanReference | None,
    operation: str,
    code: str,
) -> str:
    if not isinstance(value, str):
        raise error_type(
            f"{field_name} must be a string.",
            reference=reference,
            operation=operation,
            code=code,
        )
    normalized = value.strip().lower()
    if not normalized:
        raise error_type(
            f"{field_name} cannot be empty.",
            reference=reference,
            operation=operation,
            code=code,
        )
    if len(normalized) > max_length:
        raise error_type(
            f"{field_name} exceeds the length limit.",
            reference=reference,
            operation=operation,
            code=code,
        )
    if "/" in normalized or "\\" in normalized or ":" in normalized or ".." in normalized:
        raise error_type(
            f"{field_name} cannot be a path.",
            reference=reference,
            operation=operation,
            code=code,
        )
    if any(ord(character) < 32 for character in normalized):
        raise error_type(
            f"{field_name} cannot contain control characters.",
            reference=reference,
            operation=operation,
            code=code,
        )
    if pattern.fullmatch(normalized) is None:
        raise error_type(
            f"{field_name} has unsupported characters.",
            reference=reference,
            operation=operation,
            code=code,
        )
    return normalized


def _validate_tags(
    tags: tuple[str, ...],
    reference: ExecutionPlanReference | None,
) -> tuple[str, ...]:
    if not isinstance(tags, tuple):
        raise InvalidWorkflowTagError(
            "Workflow tags must be a tuple.",
            reference=reference,
            operation="validate_workflow_tags",
            code="INVALID_WORKFLOW_TAG",
        )
    if len(tags) > MAX_WORKFLOW_TAGS:
        raise InvalidWorkflowTagError(
            "Workflow tags exceed the tag count limit.",
            reference=reference,
            operation="validate_workflow_tags",
            code="INVALID_WORKFLOW_TAG",
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized_tag = _validate_identifier(
            tag,
            field_name="Workflow tag",
            max_length=MAX_TAG_LENGTH,
            pattern=_TAG_PATTERN,
            error_type=InvalidWorkflowTagError,
            reference=reference,
            operation="validate_workflow_tags",
            code="INVALID_WORKFLOW_TAG",
        )
        if normalized_tag in seen:
            raise InvalidWorkflowTagError(
                "Workflow tags cannot contain duplicates.",
                reference=reference,
                operation="validate_workflow_tags",
                code="DUPLICATE_WORKFLOW_TAG",
            )
        seen.add(normalized_tag)
        normalized.append(normalized_tag)
    return tuple(normalized)


def _validate_plan_structure(
    plan: ExecutionPlan,
    reference: ExecutionPlanReference,
) -> None:
    if not isinstance(plan.ordered_steps, tuple) or not plan.ordered_steps:
        raise InvalidWorkflowDefinitionError(
            "Workflow plan must contain ordered steps.",
            reference=reference,
            operation="validate_workflow_plan",
            code="INVALID_WORKFLOW_PLAN",
        )
    if any(not isinstance(step, ExecutionStep) for step in plan.ordered_steps):
        raise InvalidWorkflowDefinitionError(
            "Workflow plan ordered steps must contain ExecutionStep instances.",
            reference=reference,
            operation="validate_workflow_plan",
            code="INVALID_WORKFLOW_PLAN",
        )
    if plan.estimated_steps != len(plan.ordered_steps):
        raise InvalidWorkflowDefinitionError(
            "Workflow plan estimated_steps must match ordered steps.",
            reference=reference,
            operation="validate_workflow_plan",
            code="INVALID_WORKFLOW_PLAN",
        )
    if not isinstance(plan.required_tools, tuple):
        raise InvalidWorkflowDefinitionError(
            "Workflow plan required_tools must be a tuple.",
            reference=reference,
            operation="validate_workflow_plan",
            code="INVALID_WORKFLOW_PLAN",
        )
    if not isinstance(plan.detected_risks, tuple):
        raise InvalidWorkflowDefinitionError(
            "Workflow plan detected_risks must be a tuple.",
            reference=reference,
            operation="validate_workflow_plan",
            code="INVALID_WORKFLOW_PLAN",
        )
    if not isinstance(plan.requires_confirmation, bool):
        raise InvalidWorkflowDefinitionError(
            "Workflow plan requires_confirmation must be a bool.",
            reference=reference,
            operation="validate_workflow_plan",
            code="INVALID_WORKFLOW_PLAN",
        )


def _registry_entry_for(
    registry: ExecutionPlanRegistry,
    reference: ExecutionPlanReference,
) -> RegisteredExecutionPlan | None:
    for entry in registry.list_entries():
        if entry.reference == reference:
            return entry
    return None
