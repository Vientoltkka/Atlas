from __future__ import annotations

from pathlib import Path

from bootstrap.bootstrap import Bootstrap
from tools import atlas_manual as atlas_manual_cli
from use_cases.atlas_manual import (
    AtlasManualIndex,
    AtlasManualLoader,
    AtlasManualValidator,
    ManualLoadStatus,
    ManualSection,
    default_manual_index,
)


def test_manual_index_loads_with_unique_ids_and_stable_order() -> None:
    index = default_manual_index()
    ids = [section.id for section in index.sections]
    orders = [section.order for section in index.sections]

    assert ids == [
        "overview",
        "architecture",
        "capabilities",
        "tools",
        "execution_flow",
        "confirmations",
        "conversation",
        "operation",
        "troubleshooting",
        "limitations",
        "roadmap",
    ]
    assert len(ids) == len(set(ids))
    assert orders == sorted(orders)


def test_loader_lists_sections_and_loads_overview_architecture_unicode() -> None:
    loader = AtlasManualLoader()

    sections = loader.list_sections()
    overview = loader.get_section("overview")
    architecture = loader.get_section("architecture")

    assert sections[0].id == "overview"
    assert overview.status is ManualLoadStatus.FOUND
    assert overview.content is not None
    assert "Qué es Atlas" in overview.content
    assert architecture.status is ManualLoadStatus.FOUND
    assert architecture.content is not None
    assert "-> `ExecutionConversationController`" in architecture.content


def test_unknown_section_returns_uniform_not_found() -> None:
    result = AtlasManualLoader().get_section("missing")

    assert result.status is ManualLoadStatus.NOT_FOUND
    assert result.section is None
    assert result.content is None


def test_loading_manual_does_not_modify_content() -> None:
    loader = AtlasManualLoader()
    section = loader.list_sections()[0]
    before = section.path.read_text(encoding="utf-8-sig")

    loaded = loader.load_content(section)
    after = section.path.read_text(encoding="utf-8-sig")

    assert loaded.status is ManualLoadStatus.FOUND
    assert before == after


def test_search_matches_title_tag_and_is_case_insensitive() -> None:
    loader = AtlasManualLoader()

    assert [section.id for section in loader.search("arquitectura")] == ["architecture"]
    assert [section.id for section in loader.search("CONFIRMATION")] == ["confirmations"]
    assert loader.search("sin-resultados") == ()


def test_validator_accepts_current_manual() -> None:
    result = AtlasManualValidator().validate()

    assert result.valid is True
    assert result.issues == ()


def test_validator_detects_missing_path_and_duplicate_id(tmp_path: Path) -> None:
    existing = tmp_path / "a.md"
    existing.write_text("manual-id: same\n\nPropósito: test\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    index = AtlasManualIndex(
        sections=(
            ManualSection("same", "A", existing, "A", ("tag",), 10),
            ManualSection("same", "B", missing, "B", ("tag",), 20),
        )
    )

    result = AtlasManualValidator(loader=AtlasManualLoader(index=index)).validate()
    codes = {issue.code for issue in result.issues}

    assert result.valid is False
    assert "duplicate_id" in codes
    assert "missing_path" in codes


def test_validator_detects_missing_required_sections(tmp_path: Path) -> None:
    file_path = tmp_path / "overview.md"
    file_path.write_text("manual-id: overview\n\nPropósito: test\n", encoding="utf-8")
    index = AtlasManualIndex(
        sections=(ManualSection("overview", "Overview", file_path, "test", ("tag",), 10),)
    )

    result = AtlasManualValidator(loader=AtlasManualLoader(index=index)).validate()

    assert result.valid is False
    assert any(issue.code == "missing_required_section" for issue in result.issues)


def test_validator_detects_unknown_tool_documented(tmp_path: Path) -> None:
    manual_root = _copy_minimal_manual(tmp_path)
    tools_path = manual_root / "tools.md"
    text = tools_path.read_text(encoding="utf-8")
    text += "\n| `fake.tool` | fake | Fake | sin schema conversacional registrado | NO | fake | fake | fake |\n"
    tools_path.write_text(text, encoding="utf-8")

    result = AtlasManualValidator(
        loader=AtlasManualLoader(root=manual_root),
    ).validate()

    assert result.valid is False
    assert any(issue.code == "unknown_tool" for issue in result.issues)


def test_validator_detects_missing_registered_tool(tmp_path: Path) -> None:
    manual_root = _copy_minimal_manual(tmp_path)
    tools_path = manual_root / "tools.md"
    text = tools_path.read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if "`read_file`" not in line
    )
    tools_path.write_text(text, encoding="utf-8")

    result = AtlasManualValidator(loader=AtlasManualLoader(root=manual_root)).validate()

    assert result.valid is False
    assert any(issue.code == "missing_tool" for issue in result.issues)


def test_validator_detects_confirmation_and_schema_mismatch(tmp_path: Path) -> None:
    manual_root = _copy_minimal_manual(tmp_path)
    tools_path = manual_root / "tools.md"
    text = tools_path.read_text(encoding="utf-8")
    text = text.replace("| `write_file` | archivos | Write a UTF-8 text file. | req:path,content | SI |", "| `write_file` | archivos | Write a UTF-8 text file. | req:path | NO |")
    tools_path.write_text(text, encoding="utf-8")

    result = AtlasManualValidator(loader=AtlasManualLoader(root=manual_root)).validate()
    codes = {issue.code for issue in result.issues}

    assert result.valid is False
    assert "confirmation_mismatch" in codes
    assert "schema_mismatch" in codes


def test_validator_detects_unknown_capability_state(tmp_path: Path) -> None:
    manual_root = _copy_minimal_manual(tmp_path)
    capabilities = manual_root / "capabilities.md"
    capabilities.write_text(
        capabilities.read_text(encoding="utf-8").replace("IMPLEMENTED", "DONE", 1),
        encoding="utf-8",
    )

    result = AtlasManualValidator(loader=AtlasManualLoader(root=manual_root)).validate()

    assert result.valid is False
    assert any(issue.code == "unknown_capability_state" for issue in result.issues)


def test_documented_tools_match_registry_and_schema_sources() -> None:
    registry = Bootstrap.build_tool_registry()
    schemas = Bootstrap.build_argument_schema_registry()
    selector = Bootstrap.build_tool_selector(registry)
    tools_content = Path("docs/manual/tools.md").read_text(encoding="utf-8")

    for descriptor in registry.descriptors():
        assert f"`{descriptor.name}`" in tools_content
        expected = "| SI |" if descriptor.requires_confirmation else "| NO |"
        line = next(line for line in tools_content.splitlines() if f"`{descriptor.name}`" in line)
        assert expected in line

    for intent in selector.supported_intents():
        selection = selector.select(__import__("tools.intent_selector", fromlist=["ToolIntent"]).ToolIntent(intent))
        schema = schemas.get(intent)
        for field in schema.fields:
            line = next(line for line in tools_content.splitlines() if f"`{selection.tool_name}`" in line)
            assert field.name in line


def test_manual_module_does_not_import_executor_or_use_eval_exec() -> None:
    source = Path("use_cases/atlas_manual.py").read_text(encoding="utf-8")

    assert "ToolExecutor" not in source
    assert "eval(" not in source
    assert "exec(" not in source


def test_cli_list_show_search_validate_and_errors(capsys) -> None:
    assert atlas_manual_cli.main(["list"]) == 0
    assert "overview" in capsys.readouterr().out

    assert atlas_manual_cli.main(["show", "overview"]) == 0
    assert "Qué es Atlas" in capsys.readouterr().out

    assert atlas_manual_cli.main(["show", "missing"]) == 1
    assert "No existe" in capsys.readouterr().out

    assert atlas_manual_cli.main(["search", "confirmation"]) == 0
    assert "confirmations" in capsys.readouterr().out

    assert atlas_manual_cli.main(["search", "nope-nope"]) == 1
    assert "No hay resultados" in capsys.readouterr().out

    assert atlas_manual_cli.main(["validate"]) == 0
    assert "Manual valido" in capsys.readouterr().out

    assert atlas_manual_cli.main(["unknown"]) == 2


def _copy_minimal_manual(tmp_path: Path) -> Path:
    target = tmp_path / "manual"
    target.mkdir()
    for section in default_manual_index().sections:
        target_file = target / section.path.name
        target_file.write_text(
            section.path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return target
