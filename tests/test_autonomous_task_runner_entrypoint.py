from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.autonomous_task_runner import (
    MAX_AUTONOMOUS_ITERATIONS,
    AutonomousFileChange,
    AutonomousPlan,
    AutonomousRunnerStatus,
    EntrypointComponents,
    WORKER_ROLE_ENV,
    _build_argument_parser,
    _execute,
    _result_payload,
    main,
)
from core.autonomous_task_runner import AutonomousTaskResult
from core.model_manager import ModelSelectionRequest, ModelSelectionResult
from core.test_runner import TestRunResult


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
            output_tail="",
            command=("pytest", *test_paths),
            basetemp=None,
        )


class ScriptedPlanner:
    def __init__(self, plans: list[AutonomousPlan]) -> None:
        self._plans = list(plans)
        self.calls: list[tuple[str, int]] = []

    def __call__(self, goal: str, iteration: int, history: object) -> AutonomousPlan:
        self.calls.append((goal, iteration))
        return self._plans.pop(0)


class _EmptyModelManager:
    def resolve_model(self, identifier: str) -> None:
        return None

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResult:
        return ModelSelectionResult(
            success=False,
            logical_model_id=None,
            physical_model_name=None,
            provider_id=None,
            reason="empty registry",
            is_fallback=False,
            descriptor=None,
        )


def _args(tmp_path: Path, *extra: str):
    return _build_argument_parser().parse_args(
        [
            "hacer que los tests pasen",
            "--allowed-paths",
            "pkg",
            "--test-paths",
            "tests/test_pkg.py",
            "--project-root",
            str(tmp_path),
            *extra,
        ]
    )


def _sandbox(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")


def test_main_requires_allowed_paths_and_test_paths() -> None:
    with pytest.raises(SystemExit) as missing_allowed:
        _build_argument_parser().parse_args(
            ["meta", "--test-paths", "tests/test_x.py"]
        )
    assert missing_allowed.value.code == 2

    with pytest.raises(SystemExit) as missing_tests:
        _build_argument_parser().parse_args(["meta", "--allowed-paths", "pkg"])
    assert missing_tests.value.code == 2


def test_main_rejects_iteration_cap_breach_with_structured_error(
    tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "meta",
            "--allowed-paths",
            "pkg",
            "--test-paths",
            "tests/test_pkg.py",
            "--max-iterations",
            str(MAX_AUTONOMOUS_ITERATIONS + 1),
            "--project-root",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == AutonomousRunnerStatus.BLOCKED.value
    assert payload["reason"].startswith("invalid_config:")


def test_main_rejects_out_of_scope_allowed_path(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "meta",
            "--allowed-paths",
            "../outside",
            "--test-paths",
            "tests/test_pkg.py",
            "--project-root",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["reason"].startswith("invalid_config:")


def test_main_rejects_secret_scope(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "meta",
            "--allowed-paths",
            ".env",
            "--test-paths",
            "tests/test_pkg.py",
            "--project-root",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["reason"].startswith("invalid_config:")


def test_main_rejects_missing_project_root(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "meta",
            "--allowed-paths",
            "pkg",
            "--test-paths",
            "tests/test_pkg.py",
            "--project-root",
            str(tmp_path / "nope"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["reason"].startswith("project_root_does_not_exist")


def test_main_reports_worker_unavailability_as_error(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.delenv(WORKER_ROLE_ENV, raising=False)
    code = _execute(
        _args(tmp_path),
        project_root=tmp_path,
        model_manager=_EmptyModelManager(),
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["reason"].startswith("worker_unavailable:")


def test_execute_success_emits_structured_result_and_exit_zero(
    tmp_path: Path, capsys
) -> None:
    _sandbox(tmp_path)
    planner = ScriptedPlanner(
        [
            AutonomousPlan(
                reasoning="asignar VALUE = 2",
                changes=(AutonomousFileChange("pkg/mod.py", "VALUE = 2\n"),),
            )
        ]
    )
    tests = FakeTestRunner([True])
    code = _execute(
        _args(tmp_path, "--max-iterations", "2"),
        project_root=tmp_path,
        components=EntrypointComponents(
            planner=planner, reviewer=None, test_runner=tests
        ),
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == AutonomousRunnerStatus.SUCCESS.value
    assert payload["last_test_passed"] is True
    assert payload["iterations"][0]["outcome"] == "SUCCESS"
    assert payload["iterations"][0]["changed_paths"] == ["pkg/mod.py"]
    assert tests.calls == [("tests/test_pkg.py",)]
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_execute_blocks_changes_outside_allowed_scope(tmp_path: Path, capsys) -> None:
    _sandbox(tmp_path)
    planner = ScriptedPlanner(
        [
            AutonomousPlan(
                reasoning="intento fuera de alcance",
                changes=(AutonomousFileChange("secrets/other.py", "X = 1\n"),),
            )
        ]
    )
    code = _execute(
        _args(tmp_path),
        project_root=tmp_path,
        components=EntrypointComponents(
            planner=planner, reviewer=None, test_runner=FakeTestRunner([True])
        ),
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == AutonomousRunnerStatus.BLOCKED.value
    assert payload["reason"].startswith("out_of_scope:")
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_execute_blocks_when_iterations_exhausted(tmp_path: Path, capsys) -> None:
    _sandbox(tmp_path)
    planner = ScriptedPlanner(
        [
            AutonomousPlan(
                reasoning="cambio insuficiente",
                changes=(AutonomousFileChange("pkg/mod.py", "VALUE = 3\n"),),
            )
        ]
    )
    code = _execute(
        _args(tmp_path, "--max-iterations", "1"),
        project_root=tmp_path,
        components=EntrypointComponents(
            planner=planner, reviewer=None, test_runner=FakeTestRunner([False])
        ),
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == AutonomousRunnerStatus.BLOCKED.value
    assert payload["reason"] == "max_iterations_reached"
    assert payload["iterations"][0]["outcome"] == "RETRY"
    assert payload["iterations"][0]["restored"] is True
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_result_payload_exposes_audit_fields() -> None:
    result = AutonomousTaskResult(
        status=AutonomousRunnerStatus.SUCCESS,
        reason="goal_verified",
    )
    payload = _result_payload(result)
    assert payload["status"] == "SUCCESS"
    assert payload["reason"] == "goal_verified"
    assert payload["iterations"] == []
    assert payload["checkpoint_events"] == []
    assert payload["last_test_passed"] is None
