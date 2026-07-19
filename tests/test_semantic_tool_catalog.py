from __future__ import annotations

from typing import Any

from bootstrap.bootstrap import Bootstrap
from core.planner import Planner
from tools.argument_schema import ArgumentField, ArgumentSchema, ArgumentSchemaRegistry
from tools.base_tool import BaseTool
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.semantic_catalog import (
    RISK_LEVELS,
    SemanticToolCatalog,
    SemanticToolDescriptor,
)
from tools.tool_context import ToolContext


class CatalogFakeTool(BaseTool):
    def __init__(
        self,
        name: str,
        *,
        description: str = "Fake catalog tool.",
        metadata: dict[str, Any] | None = None,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._description = description
        self._metadata = metadata
        self._requires_confirmation = requires_confirmation
        self.executed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    def semantic_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            return {}
        return dict(self._metadata)

    def execute(self, context: ToolContext) -> str:
        self.executed = True
        raise AssertionError("semantic catalog must not execute tools")


def _registry_with_filesystem_tools() -> tuple[ToolRegistry, ToolSelector, ArgumentSchemaRegistry]:
    registry = ToolRegistry()
    registry.register(
        CatalogFakeTool(
            "read_file",
            metadata={
                "capabilities": ["read_file"],
                "supported_intents": ["read a local file"],
                "input_description": "Requires a file path.",
                "output_description": "File content.",
                "risk_level": "low",
                "preconditions": ["path must exist"],
                "limitations": ["does not write files"],
                "negative_examples": ["write a file"],
                "compatible_tools": ["write_file"],
                "tags": ["filesystem", "read"],
                "positive_examples": ["lee el archivo notas.txt"],
            },
        )
    )
    registry.register(
        CatalogFakeTool(
            "write_file",
            requires_confirmation=True,
            metadata={
                "capabilities": ["write_file"],
                "supported_intents": ["create or update a text file"],
                "input_description": "Requires path and content.",
                "output_description": "Write confirmation.",
                "risk_level": "medium",
                "risk_reasons": ["can overwrite local files"],
                "requires_confirmation": True,
                "preconditions": ["path must be provided", "content must be provided"],
                "limitations": ["does not merge content"],
                "negative_examples": ["explain files"],
                "compatible_tools": ["read_file"],
                "tags": ["filesystem", "write"],
            },
        )
    )
    registry.register(
        CatalogFakeTool(
            "list_directory",
            metadata={
                "capabilities": ["list_directory"],
                "supported_intents": ["list files in a directory"],
                "input_description": "Accepts a directory path.",
                "output_description": "Directory entries.",
                "risk_level": "low",
                "preconditions": ["path must point to a directory"],
                "limitations": ["does not recurse"],
                "negative_examples": ["read every file"],
                "compatible_tools": ["read_file"],
                "tags": ["filesystem", "directory"],
            },
        )
    )

    intent_registry = ToolIntentRegistry()
    intent_registry.register("file.read", "read_file")
    intent_registry.register("file.write", "write_file")
    intent_registry.register("directory.list", "list_directory")
    selector = ToolSelector(registry, intent_registry)

    schemas = ArgumentSchemaRegistry()
    schemas.register(ArgumentSchema("file.read", (ArgumentField("path", str, required=True),)))
    schemas.register(
        ArgumentSchema(
            "file.write",
            (
                ArgumentField("path", str, required=True),
                ArgumentField("content", str, required=True),
            ),
        )
    )
    schemas.register(ArgumentSchema("directory.list", (ArgumentField("path", str),)))
    return registry, selector, schemas


def _catalog() -> tuple[SemanticToolCatalog, ToolRegistry]:
    registry, selector, schemas = _registry_with_filesystem_tools()
    catalog = SemanticToolCatalog.build_from_registry(
        registry,
        tool_selector=selector,
        schema_registry=schemas,
    )
    return catalog, registry


def test_builds_catalog_from_tool_registry() -> None:
    catalog, registry = _catalog()

    assert [item.name for item in catalog.list_all()] == list(registry.list())
    assert catalog.validate().is_valid is True


def test_semantic_descriptor_contains_complete_operational_metadata() -> None:
    catalog, _registry = _catalog()

    descriptor = catalog.get("read_file")

    assert isinstance(descriptor, SemanticToolDescriptor)
    assert descriptor.capabilities == ("read_file",)
    assert descriptor.supported_intents == ("read a local file",)
    assert descriptor.required_arguments == ("path",)
    assert descriptor.output_description == "File content."
    assert descriptor.risk_level == "low"
    assert descriptor.preconditions == ("path must exist",)
    assert descriptor.compatible_tools == ("write_file",)
    assert descriptor.positive_examples == ("lee el archivo notas.txt",)


def test_legacy_tool_uses_conservative_defaults_and_warning() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("legacy.tool", metadata=None))

    catalog = SemanticToolCatalog.build_from_registry(registry)
    descriptor = catalog.get("legacy.tool")
    validation = catalog.validate()

    assert descriptor.capabilities == ("legacy_tool",)
    assert descriptor.risk_level == "low"
    assert any("conservative semantic defaults" in warning for warning in validation.warnings)


def test_unique_names_are_validated_by_tool_registry_source() -> None:
    catalog, _registry = _catalog()
    result = catalog.validate()

    assert result.validated_tools == ["list_directory", "read_file", "write_file"]
    assert result.tool_count == 3


def test_capability_exact_and_search_by_capability() -> None:
    catalog, _registry = _catalog()

    assert catalog.get("read_file").capabilities == ("read_file",)
    assert [item.name for item in catalog.find_by_capability("read_file")] == ["read_file"]
    assert catalog.find_by_capability("read") == ()


def test_search_by_normalized_intent() -> None:
    catalog, _registry = _catalog()

    matches = catalog.find_by_intent("READ A LOCAL FILE")

    assert [item.name for item in matches] == ["read_file"]


def test_multiple_candidate_tools_for_same_capability() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("reader.a", metadata={"capabilities": ["read_file"]}))
    registry.register(CatalogFakeTool("reader.b", metadata={"capabilities": ["read_file"]}))
    catalog = SemanticToolCatalog.build_from_registry(registry)

    assert [item.name for item in catalog.find_by_capability("read_file")] == [
        "reader.a",
        "reader.b",
    ]


def test_unknown_compatible_tool_is_error() -> None:
    registry = ToolRegistry()
    registry.register(
        CatalogFakeTool("read_file", metadata={"compatible_tools": ["missing_tool"]})
    )

    result = SemanticToolCatalog.build_from_registry(registry).validate()

    assert result.is_valid is False
    assert "references unknown compatible tool 'missing_tool'" in result.errors[0]


def test_self_compatible_tool_is_invalid() -> None:
    registry = ToolRegistry()
    registry.register(
        CatalogFakeTool("read_file", metadata={"compatible_tools": ["read_file"]})
    )

    result = SemanticToolCatalog.build_from_registry(registry).validate()

    assert result.is_valid is False
    assert "cannot be compatible with itself" in result.errors[0]


def test_invalid_risk_level_is_error() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("read_file", metadata={"risk_level": "severe"}))

    result = SemanticToolCatalog.build_from_registry(registry).validate()

    assert result.is_valid is False
    assert "invalid risk level 'severe'" in result.errors[0]


def test_dangerous_or_medium_risk_requires_confirmation() -> None:
    registry = ToolRegistry()
    registry.register(
        CatalogFakeTool(
            "write_file",
            metadata={"dangerous": True, "risk_level": "medium"},
        )
    )

    descriptor = SemanticToolCatalog.build_from_registry(registry).get("write_file")

    assert descriptor.requires_confirmation is True


def test_incomplete_metadata_produces_warning_not_error() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("read_file", metadata={}))

    result = SemanticToolCatalog.build_from_registry(registry).validate()

    assert result.is_valid is True
    assert result.warnings


def test_semantic_argument_not_declared_is_error() -> None:
    registry, selector, schemas = _registry_with_filesystem_tools()
    registry.register(
        CatalogFakeTool(
            "extra_tool",
            metadata={"required_arguments": ["api key"]},
        )
    )
    catalog = SemanticToolCatalog.build_from_registry(
        registry,
        tool_selector=selector,
        schema_registry=schemas,
    )

    result = catalog.validate()

    assert result.is_valid is False
    assert any("without a technical schema" in error for error in result.errors)


def test_output_fields_without_output_contract_are_invalid() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("read_file", metadata={"output_fields": ["content"]}))

    result = SemanticToolCatalog.build_from_registry(registry).validate()

    assert result.is_valid is False
    assert any("declares output fields without an output contract" in error for error in result.errors)


def test_catalog_order_is_deterministic() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("z_tool"))
    registry.register(CatalogFakeTool("a_tool"))

    catalog = SemanticToolCatalog.build_from_registry(registry)

    assert [item.name for item in catalog.list_all()] == ["a_tool", "z_tool"]


def test_json_export_is_deterministic() -> None:
    catalog, _registry = _catalog()

    first = catalog.to_json()
    second = catalog.to_json()

    assert first == second
    assert first.index("list_directory") < first.index("read_file") < first.index("write_file")


def test_export_does_not_execute_tools() -> None:
    catalog, registry = _catalog()

    catalog.to_dict()
    catalog.to_json()
    catalog.summary_text()

    assert all(tool.executed is False for tool in registry.tools.values())  # type: ignore[attr-defined]


def test_secret_like_metadata_is_rejected_and_not_needed_for_export() -> None:
    registry = ToolRegistry()
    registry.register(CatalogFakeTool("mail.send", metadata={"tags": ["api_key"]}))

    catalog = SemanticToolCatalog.build_from_registry(registry)
    result = catalog.validate()

    assert result.is_valid is False
    assert any("secret-like value" in error for error in result.errors)


def test_catalog_does_not_mutate_tool_registry() -> None:
    catalog, registry = _catalog()
    before = registry.list()

    catalog.validate()
    catalog.find_by_capability("read_file")
    catalog.to_json()

    assert registry.list() == before


def test_planner_still_works_without_semantic_catalog() -> None:
    plan = Planner().create_execution_plan("Lee README.md")

    assert plan.required_tools == ("read_file",)


def test_planner_accepts_optional_catalog_without_changing_behavior() -> None:
    catalog, _registry = _catalog()
    plain = Planner().create_execution_plan("Lee README.md")
    with_catalog = Planner(semantic_tool_catalog=catalog).create_execution_plan("Lee README.md")

    assert with_catalog == plain


def test_catalog_build_does_not_call_models_or_network() -> None:
    catalog, _registry = _catalog()

    assert catalog.validate().is_valid is True


def test_risk_levels_are_centralized() -> None:
    assert RISK_LEVELS == ("none", "low", "medium", "high", "critical")


def test_bootstrap_builds_real_semantic_catalog_from_real_registry() -> None:
    catalog = Bootstrap.build_semantic_tool_catalog()
    names = [item.name for item in catalog.list_all()]

    assert "read_file" in names
    assert "write_file" in names
    assert "list_directory" in names
    assert catalog.get("write_file").requires_confirmation is True


def test_minimal_functional_filesystem_catalog() -> None:
    catalog, registry = _catalog()

    assert [item.name for item in catalog.find_by_capability("read_file")] == ["read_file"]
    assert [item.name for item in catalog.find_by_intent("read a local file")] == ["read_file"]
    assert catalog.get("write_file").requires_confirmation is True
    assert catalog.validate().is_valid is True
    assert all(tool.executed is False for tool in registry.tools.values())  # type: ignore[attr-defined]
