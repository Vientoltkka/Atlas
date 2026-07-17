import pytest

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaAlreadyRegisteredError,
    ArgumentSchemaNotRegisteredError,
    ArgumentSchemaRegistry,
    ArgumentValidationError,
    ArgumentValidator,
    require_non_empty,
)
from tools.base_tool import BaseTool
from tools.intent_selector import ToolIntent, ToolSelection
from tools.registry import ToolDescriptor, ToolRegistry
from tools.tool_context import ToolContext


class ExplodingTool(BaseTool):
    def __init__(self) -> None:
        self.executed = False

    @property
    def name(self) -> str:
        return "demo.exploding"

    @property
    def description(self) -> str:
        return "Explodes if executed."

    def execute(self, context: ToolContext):
        self.executed = True
        raise AssertionError("Validation must not execute tools.")


def _selection(
    action: str,
    arguments: dict,
    tool_name: str = "demo.tool",
) -> ToolSelection:
    return ToolSelection(
        intent=ToolIntent(action=action, arguments=arguments),
        tool_name=tool_name,
        descriptor=ToolDescriptor(
            name=tool_name,
            description="Demo tool.",
            tool=object(),
        ),
        arguments=arguments,
    )


def test_argument_schema_registry_registers_gets_lists_and_checks_exists() -> None:
    registry = ArgumentSchemaRegistry()
    schema = ArgumentSchema(
        "demo.action",
        (ArgumentField("path", str, required=True),),
    )

    registry.register(schema)

    assert registry.exists("demo.action") is True
    assert registry.get("demo.action") is schema
    assert registry.list() == ("demo.action",)


def test_argument_schema_registry_rejects_duplicate_intents() -> None:
    registry = ArgumentSchemaRegistry()
    schema = ArgumentSchema("demo.action")
    registry.register(schema)

    with pytest.raises(ArgumentSchemaAlreadyRegisteredError, match="already registered"):
        registry.register(schema)


def test_argument_schema_registry_reports_missing_schema_clearly() -> None:
    registry = ArgumentSchemaRegistry()

    with pytest.raises(ArgumentSchemaNotRegisteredError, match="not registered"):
        registry.get("missing.action")


def test_argument_schema_registry_exposes_read_only_collection() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(ArgumentSchema("demo.action"))

    with pytest.raises(TypeError):
        registry.schemas["other.action"] = ArgumentSchema("other.action")

    assert registry.exists("other.action") is False


def test_argument_schema_rejects_duplicate_fields() -> None:
    with pytest.raises(ValueError, match="Duplicate argument field"):
        ArgumentSchema(
            "demo.action",
            (
                ArgumentField("path", str),
                ArgumentField("path", str),
            ),
        )


def test_validator_accepts_required_argument_and_returns_normalized_mapping() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(
        ArgumentSchema(
            "file.read",
            (ArgumentField("path", str, required=True),),
        )
    )
    validator = ArgumentValidator(registry)

    result = validator.validate(_selection("file.read", {"path": "README.md"}, "read_file"))

    assert result.valid is True
    assert result.executed is False
    assert result.tool_name == "read_file"
    assert dict(result.original_arguments) == {"path": "README.md"}
    assert dict(result.validated_arguments) == {"path": "README.md"}


def test_validator_applies_defaults_without_mutating_original_arguments() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(
        ArgumentSchema(
            "directory.list",
            (ArgumentField("path", str, default="."),),
        )
    )
    validator = ArgumentValidator(registry)
    arguments: dict = {}

    result = validator.validate(_selection("directory.list", arguments, "list_directory"))

    assert arguments == {}
    assert dict(result.original_arguments) == {}
    assert dict(result.validated_arguments) == {"path": "."}


def test_validator_rejects_missing_required_argument() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(
        ArgumentSchema(
            "file.read",
            (ArgumentField("path", str, required=True),),
        )
    )
    validator = ArgumentValidator(registry)

    with pytest.raises(ArgumentValidationError) as raised:
        validator.validate(_selection("file.read", {}, "read_file"))

    assert raised.value.intent_action == "file.read"
    assert raised.value.field == "path"
    assert raised.value.reason == "required argument is missing"


def test_validator_rejects_unknown_arguments() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(ArgumentSchema("file.read", (ArgumentField("path", str),)))
    validator = ArgumentValidator(registry)

    with pytest.raises(ArgumentValidationError) as raised:
        validator.validate(_selection("file.read", {"unknown": "value"}, "read_file"))

    assert raised.value.field == "unknown"
    assert raised.value.reason == "unexpected argument"


def test_validator_rejects_wrong_type_and_none_when_not_allowed() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(
        ArgumentSchema(
            "file.write",
            (
                ArgumentField("path", str, required=True),
                ArgumentField("content", str, required=True),
            ),
        )
    )
    validator = ArgumentValidator(registry)

    with pytest.raises(ArgumentValidationError, match="expected str, got int"):
        validator.validate(_selection("file.write", {"path": 123, "content": "x"}))

    with pytest.raises(ArgumentValidationError, match="None is not allowed"):
        validator.validate(_selection("file.write", {"path": "x.txt", "content": None}))


def test_validator_allows_none_only_when_declared() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(
        ArgumentSchema(
            "demo.optional",
            (ArgumentField("note", str, required=True, allow_none=True),),
        )
    )
    validator = ArgumentValidator(registry)

    result = validator.validate(_selection("demo.optional", {"note": None}))

    assert dict(result.validated_arguments) == {"note": None}


def test_validator_rejects_empty_values_with_custom_validator() -> None:
    registry = ArgumentSchemaRegistry()
    registry.register(
        ArgumentSchema(
            "desktop.hotkey.press",
            (ArgumentField("keys", list, required=True, validator=require_non_empty),),
        )
    )
    validator = ArgumentValidator(registry)

    with pytest.raises(ArgumentValidationError, match="value cannot be empty"):
        validator.validate(_selection("desktop.hotkey.press", {"keys": []}))


def test_validator_does_not_execute_registered_tool() -> None:
    tool = ExplodingTool()
    tool_registry = ToolRegistry()
    tool_registry.register(tool)
    schema_registry = ArgumentSchemaRegistry()
    schema_registry.register(
        ArgumentSchema(
            "demo.explode",
            (ArgumentField("path", str, required=True),),
        )
    )
    validator = ArgumentValidator(schema_registry)

    result = validator.validate(
        _selection("demo.explode", {"path": "README.md"}, "demo.exploding")
    )

    assert result.executed is False
    assert tool.executed is False


@pytest.mark.parametrize(
    ("action", "arguments", "expected"),
    (
        ("file.read", {"path": "README.md"}, {"path": "README.md"}),
        ("file.write", {"path": "out.txt", "content": "hello"}, {"path": "out.txt", "content": "hello"}),
        ("directory.list", {}, {"path": "."}),
        ("project.tree", {}, {"path": "."}),
        ("desktop.application.open", {"application": "notepad"}, {"application": "notepad"}),
        ("desktop.file.open", {"path": "README.md"}, {"path": "README.md"}),
        ("desktop.text.type", {"text": "hola", "window_title": "Atlas"}, {"text": "hola", "window_title": "Atlas"}),
        ("desktop.hotkey.press", {"keys": ["ctrl", "s"], "window_title": "Atlas"}, {"keys": ["ctrl", "s"], "window_title": "Atlas"}),
        ("desktop.windows.list", {"title": "Atlas"}, {"title": "Atlas"}),
    ),
)
def test_bootstrap_argument_validator_accepts_supported_intents(
    action: str,
    arguments: dict,
    expected: dict,
) -> None:
    selector = Bootstrap.build_tool_selector()
    validator = Bootstrap.build_argument_validator()

    selection = selector.select(ToolIntent(action=action, arguments=arguments))
    result = validator.validate(selection)

    assert result.valid is True
    assert result.executed is False
    assert dict(result.validated_arguments) == expected


def test_bootstrap_argument_schemas_match_supported_selector_intents() -> None:
    schema_registry = Bootstrap.build_argument_schema_registry()
    selector = Bootstrap.build_tool_selector()

    assert schema_registry.list() == selector.supported_intents()


def test_bootstrap_validator_rejects_project_tree_max_depth_until_tool_supports_it() -> None:
    selector = Bootstrap.build_tool_selector()
    validator = Bootstrap.build_argument_validator()
    selection = selector.select(
        ToolIntent(
            action="project.tree",
            arguments={"path": ".", "max_depth": 2},
        )
    )

    with pytest.raises(ArgumentValidationError) as raised:
        validator.validate(selection)

    assert raised.value.field == "max_depth"
    assert raised.value.reason == "unexpected argument"


def test_bootstrap_validator_rejects_hotkey_string_keys() -> None:
    selector = Bootstrap.build_tool_selector()
    validator = Bootstrap.build_argument_validator()
    selection = selector.select(
        ToolIntent(
            action="desktop.hotkey.press",
            arguments={"keys": "ctrl+s", "window_title": "Atlas"},
        )
    )

    with pytest.raises(ArgumentValidationError, match="expected list, got str"):
        validator.validate(selection)
