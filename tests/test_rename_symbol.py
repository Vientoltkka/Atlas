from pathlib import Path

from domain.refactoring.refactoring_plan import RefactoringPlan
from models.architecture_graph import ArchitectureGraph, ArchitectureNode
from use_cases.plan_refactoring import PlanRefactoringUseCase
from use_cases.rename_symbol import RenameSymbolUseCase


class _StaticPlanRefactoring:
    def __init__(
        self,
        plan: RefactoringPlan,
    ) -> None:
        self._plan = plan

    def execute(
        self,
        symbol_name: str,
        new_name: str,
    ) -> RefactoringPlan:
        return self._plan


def _write(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _plan(
    affected_files: list[str],
    definition_file: str = "router.py",
    can_apply: bool = True,
    warnings: list[str] | None = None,
) -> RefactoringPlan:
    return RefactoringPlan(
        operation="rename_symbol",
        symbol_name="Router",
        new_name="RequestRouter",
        definition_file=definition_file,
        affected_files=affected_files,
        can_apply=can_apply,
        warnings=warnings or [],
    )


def test_renames_class_definition(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is True
    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "class RequestRouter:\n    pass\n"
    )


def test_renames_imported_reference(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    _write(tmp_path / "consumer.py", "from router import Router\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["router.py", "consumer.py"]))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is True
    assert (tmp_path / "consumer.py").read_text(encoding="utf-8") == (
        "from router import RequestRouter\n"
    )


def test_renames_call_or_symbol_use(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    _write(tmp_path / "main.py", "value = Router()\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["router.py", "main.py"]))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is True
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == (
        "value = RequestRouter()\n"
    )


def test_modifies_multiple_files_in_plan(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    _write(tmp_path / "a.py", "from router import Router\n")
    _write(tmp_path / "b.py", "item = Router()\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["router.py", "a.py", "b.py"]))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is True
    assert len(result.changed_files) == 3
    assert "RequestRouter" in (tmp_path / "a.py").read_text(encoding="utf-8")
    assert "RequestRouter" in (tmp_path / "b.py").read_text(encoding="utf-8")


def test_does_not_modify_comments(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "# Router comment\nclass Router:\n    pass\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    use_case.execute(tmp_path, "Router", "RequestRouter")

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "# Router comment\nclass RequestRouter:\n    pass\n"
    )


def test_does_not_modify_strings(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", 'name = "Router"\nclass Router:\n    pass\n')
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    use_case.execute(tmp_path, "Router", "RequestRouter")

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        'name = "Router"\nclass RequestRouter:\n    pass\n'
    )


def test_does_not_modify_partial_identifiers(tmp_path: Path) -> None:
    _write(
        tmp_path / "router.py",
        "class RouterConfig:\n    pass\nclass MyRouter:\n    pass\nclass Router:\n    pass\n",
    )
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    use_case.execute(tmp_path, "Router", "RequestRouter")

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "class RouterConfig:\n    pass\n"
        "class MyRouter:\n    pass\n"
        "class RequestRouter:\n    pass\n"
    )


def test_does_not_modify_case_differences(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "router = 1\nRouter = 2\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    use_case.execute(tmp_path, "Router", "RequestRouter")

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "router = 1\nRequestRouter = 2\n"
    )


def test_counts_replacements(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "Router = Router()\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.replacements_count == 2


def test_changed_files_are_deterministic(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    _write(tmp_path / "z.py", "Router()\n")
    _write(tmp_path / "a.py", "Router()\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["z.py", "router.py", "a.py"]))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.changed_files == (
        tmp_path / "a.py",
        tmp_path / "router.py",
        tmp_path / "z.py",
    )


def test_rejects_equal_names(tmp_path: Path) -> None:
    use_case = RenameSymbolUseCase(PlanRefactoringUseCase(ArchitectureGraph()))

    result = use_case.execute(tmp_path, "Router", "Router")

    assert result.applied is False
    assert result.warnings


def test_rejects_invalid_new_name(tmp_path: Path) -> None:
    use_case = RenameSymbolUseCase(PlanRefactoringUseCase(ArchitectureGraph()))

    result = use_case.execute(tmp_path, "Router", "123Router")

    assert result.applied is False
    assert result.warnings


def test_does_not_apply_when_plan_cannot_apply(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["router.py"], can_apply=False))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is False
    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "class Router:\n    pass\n"
    )


def test_rejects_missing_file(tmp_path: Path) -> None:
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["missing.py"], definition_file="missing.py"))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is False
    assert result.warnings == ("archivo inexistente: missing.py",)


def test_rejects_file_outside_project_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    _write(outside, "class Router:\n    pass\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(
            _plan([str(outside)], definition_file=str(outside))
        )
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is False
    assert result.warnings == (f"archivo fuera de project_root: {outside}",)
    outside.unlink()


def test_rejects_non_python_file(tmp_path: Path) -> None:
    _write(tmp_path / "router.txt", "Router\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["router.txt"], definition_file="router.txt"))
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is False
    assert result.warnings == ("archivo no Python: router.txt",)


def test_does_not_write_when_resulting_content_is_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(tmp_path / "router.py", "class Router:\n    pass\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    monkeypatch.setattr(
        use_case,
        "_rename_tokens",
        lambda source, symbol_name, new_name: ("class Broken(:\n", 1),
    )

    try:
        use_case.execute(tmp_path, "Router", "RequestRouter")
    except Exception:
        pass

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "class Router:\n    pass\n"
    )


def test_rolls_back_if_write_fails_after_previous_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(tmp_path / "a.py", "Router()\n")
    _write(tmp_path / "b.py", "Router()\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["a.py", "b.py"], definition_file="a.py"))
    )
    original_write = use_case._write_file

    def failing_write(path: Path, content: str) -> None:
        if path.name == "b.py" and "RequestRouter" in content:
            raise OSError("controlled write failure")

        original_write(path, content)

    monkeypatch.setattr(use_case, "_write_file", failing_write)

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is False
    assert result.rolled_back is True
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "Router()\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "Router()\n"


def test_rolls_back_if_post_write_validation_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(tmp_path / "router.py", "Router()\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    def failing_validation(changed_files: tuple[Path, ...]) -> None:
        raise SyntaxError("controlled post validation failure")

    monkeypatch.setattr(use_case, "_validate_written_files", failing_validation)

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is False
    assert result.rolled_back is True
    assert (tmp_path / "router.py").read_text(encoding="utf-8") == "Router()\n"


def test_does_not_leave_partially_modified_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(tmp_path / "a.py", "Router()\n")
    _write(tmp_path / "b.py", "Router()\n")
    use_case = RenameSymbolUseCase(
        _StaticPlanRefactoring(_plan(["a.py", "b.py"], definition_file="a.py"))
    )

    monkeypatch.setattr(
        use_case,
        "_validate_written_files",
        lambda changed_files: (_ for _ in ()).throw(SyntaxError("failed")),
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.rolled_back is True
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "Router()\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "Router()\n"


def test_returns_applied_true_when_successful(tmp_path: Path) -> None:
    _write(tmp_path / "router.py", "Router()\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is True
    assert result.rolled_back is False


def test_returns_rolled_back_true_when_restored(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "router.py", "Router()\n")
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    monkeypatch.setattr(
        use_case,
        "_validate_written_files",
        lambda changed_files: (_ for _ in ()).throw(SyntaxError("failed")),
    )

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.rolled_back is True


def test_original_content_is_preserved_after_rollback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_content = "Router()\n"
    _write(tmp_path / "router.py", original_content)
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    monkeypatch.setattr(
        use_case,
        "_validate_written_files",
        lambda changed_files: (_ for _ in ()).throw(SyntaxError("failed")),
    )

    use_case.execute(tmp_path, "Router", "RequestRouter")

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == original_content


def test_does_not_use_blind_text_replacement(tmp_path: Path) -> None:
    _write(
        tmp_path / "router.py",
        (
            "# Router\n"
            'text = "Router"\n'
            "class RouterConfig:\n    pass\n"
            "class Router:\n    pass\n"
        ),
    )
    use_case = RenameSymbolUseCase(_StaticPlanRefactoring(_plan(["router.py"])))

    use_case.execute(tmp_path, "Router", "RequestRouter")

    assert (tmp_path / "router.py").read_text(encoding="utf-8") == (
        "# Router\n"
        'text = "Router"\n'
        "class RouterConfig:\n    pass\n"
        "class RequestRouter:\n    pass\n"
    )


def test_integration_with_plan_refactoring_use_case(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py", "class Router:\n    pass\n")
    _write(tmp_path / "main.py", "from core.router import Router\nRouter()\n")
    graph = ArchitectureGraph(
        nodes=[
            ArchitectureNode(
                path="core/router.py",
                module="core.router",
                classes=["Router"],
            ),
            ArchitectureNode(
                path="main.py",
                module="main",
                imports=["core.router.Router"],
                dependencies=["core/router.py"],
            ),
        ]
    )
    use_case = RenameSymbolUseCase(PlanRefactoringUseCase(graph))

    result = use_case.execute(tmp_path, "Router", "RequestRouter")

    assert result.applied is True
    assert result.changed_files == (
        tmp_path / "core" / "router.py",
        tmp_path / "main.py",
    )
