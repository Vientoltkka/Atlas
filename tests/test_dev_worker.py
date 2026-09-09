from __future__ import annotations

from pathlib import Path

import pytest

from core.dev_worker import (
    CommandResult,
    DevStep,
    DevWorker,
    DevWorkerAction,
    SandboxViolationError,
    classify_command,
    resolve_in_workspace,
)
from core.test_runner import PytestRunner, TestRunResult


class FakeTestRunner:
    def __init__(self, results: list[bool]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, ...]] = []

    def run(self, test_paths: tuple[str, ...]) -> TestRunResult:
        self.calls.append(tuple(test_paths))
        passed = self._results.pop(0) if self._results else self._results[-1]
        return TestRunResult(
            passed=passed,
            exit_code=0 if passed else 1,
            timed_out=False,
            detail="fake",
            output_tail="" if passed else "1 failed, 0 passed",
            command=("pytest", *test_paths),
            basetemp=None,
        )


class RecordingCommandRunner:
    def __init__(self, exit_code: int = 0) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._exit_code = exit_code

    def __call__(self, command: tuple[str, ...]) -> CommandResult:
        self.calls.append(command)
        return CommandResult(
            exit_code=self._exit_code,
            stdout_tail="fake stdout",
            stderr_tail="",
        )


def _worker(tmp_path: Path, **kwargs: object) -> DevWorker:
    return DevWorker(tmp_path, **kwargs)


# A. Caso controlado: inspeccionar fixture -> edit minimo -> test focal -> DONE


def test_controlled_case_inspect_edit_test_done(tmp_path: Path) -> None:
    (tmp_path / "greeter.py").write_text('def greet(name):\n    return "BUG"\n', encoding="utf-8")
    steps = (
        DevStep(DevWorkerAction.READ, path="greeter.py", description="inspeccionar fixture"),
        DevStep(DevWorkerAction.SEARCH, pattern="def greet"),
        DevStep(
            DevWorkerAction.EDIT,
            path="greeter.py",
            operation="replace_text",
            old_text='return "BUG"',
            new_text='return "hi " + name',
            description="correccion minima",
        ),
        DevStep(DevWorkerAction.TEST, test_paths=("test_greeter.py",)),
    )
    tests = FakeTestRunner([True])

    result = _worker(tmp_path, test_runner=tests).execute("hacer que test_greeter pase", steps)

    assert result.status == "DONE"
    assert result.error is None
    assert (tmp_path / "greeter.py").read_text(encoding="utf-8") == 'def greet(name):\n    return "hi " + name\n'
    assert tests.calls == [("test_greeter.py",)]
    details = [item.detail for item in result.evidence]
    assert any("inspeccionar fixture" in detail for detail in details)
    assert any("PASS" in detail for detail in details)
    assert "STATUS: DONE" in result.report


def test_controlled_case_with_real_pytest_runner(tmp_path: Path) -> None:
    (tmp_path / "greeter.py").write_text('def greet(name):\n    return "BUG"\n', encoding="utf-8")
    (tmp_path / "test_greeter.py").write_text(
        "from greeter import greet\n\n\ndef test_greet():\n    assert greet('atlas') == 'hi atlas'\n",
        encoding="utf-8",
    )
    steps = (
        DevStep(DevWorkerAction.READ, path="greeter.py"),
        DevStep(
            DevWorkerAction.EDIT,
            path="greeter.py",
            operation="replace_text",
            old_text='return "BUG"',
            new_text='return "hi " + name',
        ),
        DevStep(DevWorkerAction.TEST, test_paths=("test_greeter.py",)),
    )

    result = DevWorker(tmp_path, test_runner=PytestRunner(tmp_path)).execute(
        "corregir greeter y verificar", steps
    )

    assert result.status == "DONE"
    assert any("PASS" in item.detail for item in result.evidence)


# B. fallo de test -> reintento limitado


def test_permanent_focal_failure_blocks_after_retry_limit(tmp_path: Path) -> None:
    steps = (DevStep(DevWorkerAction.TEST, test_paths=("tests/test_greeter.py",)),)
    tests = FakeTestRunner([False, False, False])

    result = _worker(tmp_path, test_runner=tests).execute("objetivo imposible", steps)

    assert result.status == "BLOCKED"
    assert "retry limit 2" in result.error
    assert len(tests.calls) == 3
    assert "STATUS: BLOCKED" in result.report


def test_focal_failure_recovers_within_retries(tmp_path: Path) -> None:
    steps = (
        DevStep(DevWorkerAction.EDIT, path="greeter.py", operation="full_file", content="fixed\n"),
        DevStep(DevWorkerAction.TEST, test_paths=("tests/test_greeter.py",)),
    )
    (tmp_path / "greeter.py").write_text("broken\n", encoding="utf-8")
    tests = FakeTestRunner([False, True])

    result = _worker(tmp_path, test_runner=tests).execute("corregir con reintento", steps)

    assert result.status == "DONE"
    assert any("attempt 2" in item.detail for item in result.evidence)


# C. accion sensible -> WAITING_APPROVAL


def test_git_push_waits_for_approval(tmp_path: Path) -> None:
    runner = RecordingCommandRunner()
    steps = (
        DevStep(DevWorkerAction.EDIT, path="greeter.py", operation="full_file", content="ok\n"),
        DevStep(DevWorkerAction.COMMAND, command="git push origin main"),
    )

    result = DevWorker(tmp_path, command_runner=runner).execute("publicar cambio", steps)

    assert result.status == "WAITING_APPROVAL"
    assert result.pending_action == "git push origin main"
    assert "push" in result.pending_reason
    assert runner.calls == []
    assert (tmp_path / "greeter.py").read_text(encoding="utf-8") == "ok\n"


def test_dependency_install_waits_for_approval(tmp_path: Path) -> None:
    runner = RecordingCommandRunner()
    steps = (DevStep(DevWorkerAction.COMMAND, command="pip install requests"),)

    result = DevWorker(tmp_path, command_runner=runner).execute("instalar deps", steps)

    assert result.status == "WAITING_APPROVAL"
    assert "dependencies" in result.pending_reason
    assert runner.calls == []


def test_safe_command_runs_inside_workspace(tmp_path: Path) -> None:
    runner = RecordingCommandRunner()
    steps = (DevStep(DevWorkerAction.COMMAND, command="python --version"),)

    result = DevWorker(tmp_path, command_runner=runner).execute("verificar python", steps)

    assert result.status == "DONE"
    assert runner.calls == [("python", "--version")]
    assert any("OK" in item.detail for item in result.evidence)


# D. path fuera de C:\AI\Atlas -> bloqueado


def test_edit_outside_workspace_is_blocked(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_escape.txt"
    steps = (DevStep(DevWorkerAction.EDIT, path=str(outside), operation="full_file", content="x"),)

    result = DevWorker(tmp_path).execute("salir del repo", steps)

    assert result.status == "BLOCKED"
    assert "outside the workspace root" in result.error
    assert not outside.exists()


def test_read_outside_workspace_is_blocked(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("leak\n", encoding="utf-8")
    steps = (DevStep(DevWorkerAction.READ, path=str(outside)),)

    result = DevWorker(tmp_path).execute("leer fuera", steps)

    assert result.status == "BLOCKED"
    assert "outside the workspace root" in result.error


# E. ficheros protegidos no editables


def test_protected_authorization_file_edit_is_blocked(tmp_path: Path) -> None:
    steps = (
        DevStep(
            DevWorkerAction.EDIT,
            path="core/execution_authorization.py",
            operation="replace_text",
            old_text="AUTHORIZED",
            new_text="ALWAYS_AUTHORIZED",
        ),
    )

    result = DevWorker(tmp_path).execute("auto-modificar autorizacion", steps)

    assert result.status == "BLOCKED"
    assert "protected" in result.error


def test_env_edit_is_blocked(tmp_path: Path) -> None:
    steps = (
        DevStep(DevWorkerAction.EDIT, path=".env", operation="full_file", content="X=1"),
    )

    result = DevWorker(tmp_path).execute("tocar secrets", steps)

    assert result.status == "BLOCKED"
    assert "protected" in result.error


# F. clasificador de comandos sensibles (tabla directa)


def test_classify_command_table() -> None:
    assert classify_command("git push origin main") is not None
    assert classify_command("git reset --hard") is not None
    assert classify_command("git stash") is not None
    assert classify_command("del core\\atlas.py") is not None
    assert classify_command("pip install requests") is not None
    assert classify_command("curl https://example.com") is not None
    assert classify_command("type .env") is not None
    assert classify_command("python --version") is None
    assert classify_command(("python", "-m", "pytest", "-q")) is None
    assert classify_command("") is None


def test_sandbox_helper_refuses_directly(tmp_path: Path) -> None:
    outside = tmp_path.parent / "x.txt"
    with pytest.raises(SandboxViolationError):
        resolve_in_workspace(tmp_path, str(outside))
