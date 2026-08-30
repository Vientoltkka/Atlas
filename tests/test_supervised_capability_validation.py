from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
from types import SimpleNamespace

from agents.coding_agent import CodingAgent


_ALLOWED = (
    "tools/temperature_conversion.py",
    "bootstrap/bootstrap.py",
    "tests/test_temperature_conversion.py",
)


class _Writer:
    def __init__(self, fail_on: int | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail_on = fail_on

    def execute(self, path: str, content: str) -> str:
        self.calls.append((path, content))
        if self._fail_on == len(self.calls):
            raise OSError("rollback write failed")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "written"


def _agent(tmp_path: Path, writer: _Writer) -> CodingAgent:
    agent = CodingAgent(
        prompt_client=SimpleNamespace(),
        read_file=SimpleNamespace(),
        write_file=writer,
    )
    agent._project_root = tmp_path
    proposed_contents = {}
    for relative_path in _ALLOWED:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"original:{relative_path}", encoding="utf-8")
        proposed_contents[relative_path] = f"applied:{relative_path}"
    agent.prepare_capability_plan(
        capability_id="unit.temperature-conversion",
        implementation="tool determinista",
        planned_files=_ALLOWED,
        focused_tests=("tests/test_temperature_conversion.py",),
        risk="acotado",
        proposed_contents=proposed_contents,
    )
    return agent


def _result(returncode: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("pytest",), returncode, "", "")


def test_validated_capability_requests_final_human_approval(tmp_path, monkeypatch) -> None:
    writer = _Writer()
    agent = _agent(tmp_path, writer)
    monkeypatch.setattr(agent, "_run_capability_tests", lambda: _result(0))

    response = agent.apply_prepared_capability_plan("unit.temperature-conversion")

    assert agent.capability_validation_status == "VALIDATED"
    assert "Validación completada correctamente." in response
    assert "¿Apruebas cerrar y versionar esta mejora?" in response
    assert len(writer.calls) == 3
    assert agent._pending_capability_plan is None
    assert agent._applied_capability_change is None


def test_failed_validation_rolls_back_only_the_approved_files(tmp_path, monkeypatch) -> None:
    writer = _Writer()
    agent = _agent(tmp_path, writer)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(agent, "_run_capability_tests", lambda: _result(1))

    response = agent.apply_prepared_capability_plan("unit.temperature-conversion")

    assert "Se restauraron únicamente los archivos aprobados" in response
    assert agent.capability_validation_status is None
    assert agent._pending_capability_plan is None
    assert agent._applied_capability_change is None
    assert unrelated.read_text(encoding="utf-8") == "keep"
    for relative_path in _ALLOWED:
        assert (tmp_path / relative_path).read_text(encoding="utf-8") == f"original:{relative_path}"


def test_rollback_stops_without_overwriting_external_changes(tmp_path, monkeypatch) -> None:
    writer = _Writer()
    agent = _agent(tmp_path, writer)

    def failed_tests() -> subprocess.CompletedProcess[str]:
        (tmp_path / _ALLOWED[0]).write_text("external change", encoding="utf-8")
        return _result(1)

    monkeypatch.setattr(agent, "_run_capability_tests", failed_tests)
    response = agent.apply_prepared_capability_plan("unit.temperature-conversion")

    assert "no sobrescribir cambios ajenos" in response
    assert (tmp_path / _ALLOWED[0]).read_text(encoding="utf-8") == "external change"
    assert agent._pending_capability_plan is None
    assert agent._applied_capability_change is None


def test_rollback_write_failure_reports_and_stops(tmp_path, monkeypatch) -> None:
    writer = _Writer(fail_on=4)
    agent = _agent(tmp_path, writer)
    monkeypatch.setattr(agent, "_run_capability_tests", lambda: _result(1))

    response = agent.apply_prepared_capability_plan("unit.temperature-conversion")

    assert "ERROR CRÍTICO: falló el rollback controlado" in response
    assert agent._pending_capability_plan is None
    assert agent._applied_capability_change is None


def test_invalid_scope_does_not_write_or_run_validation(tmp_path, monkeypatch) -> None:
    writer = _Writer()
    agent = _agent(tmp_path, writer)
    agent._pending_capability_plan["planned_files"] = _ALLOWED + ("README.md",)
    monkeypatch.setattr(agent, "_run_capability_tests", lambda: (_ for _ in ()).throw(AssertionError("must not validate")))

    assert "alcance del plan no es válido" in agent.apply_prepared_capability_plan("unit.temperature-conversion")
    assert writer.calls == []


def test_validation_flow_never_creates_a_commit() -> None:
    source = inspect.getsource(CodingAgent)

    assert "git commit" not in source.lower()
    assert "subprocess.run" in source
class _Git:
    def __init__(self, *, commit_returncode: int = 0, staged_names: tuple[str, ...] = _ALLOWED) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._commit_returncode = commit_returncode
        self._staged_names = staged_names

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        self.commands.append(arguments)
        if arguments[0] == "status":
            return subprocess.CompletedProcess(arguments, 0, "\n".join(" M " + name for name in self._staged_names), "")
        if arguments[0] == "commit":
            return subprocess.CompletedProcess(arguments, self._commit_returncode, "", "commit failed")
        if arguments[0] == "rev-parse":
            return subprocess.CompletedProcess(arguments, 0, "abc123\n", "")
        return subprocess.CompletedProcess(arguments, 0, "", "")


def _validated_agent(tmp_path: Path, monkeypatch) -> tuple[CodingAgent, _Writer]:
    writer = _Writer()
    agent = _agent(tmp_path, writer)
    monkeypatch.setattr(agent, "_run_capability_tests", lambda: _result(0))
    agent.apply_prepared_capability_plan("unit.temperature-conversion")
    assert agent.capability_validation_status == "VALIDATED"
    return agent, writer


def test_final_approval_commits_exactly_the_validated_scope(tmp_path, monkeypatch) -> None:
    agent, _ = _validated_agent(tmp_path, monkeypatch)
    git = _Git()
    monkeypatch.setattr(agent, "_git_command", git.run)

    response = agent.close_validated_capability_plan("unit.temperature-conversion", approved=True)

    assert "Commit: abc123" in response
    assert agent.capability_validation_status is None
    assert git.commands[0] == ("status", "--porcelain", "--", *_ALLOWED)
    assert git.commands[1][0:2] == ("commit", "--only")
    assert git.commands[1][-4:] == ("--", *_ALLOWED)
    assert all(command[0] != "push" for command in git.commands)


def test_final_rejection_creates_no_commit_or_rollback(tmp_path, monkeypatch) -> None:
    agent, writer = _validated_agent(tmp_path, monkeypatch)
    git = _Git()
    monkeypatch.setattr(agent, "_git_command", git.run)

    response = agent.close_validated_capability_plan("unit.temperature-conversion", approved=False)

    assert "no aprobado" in response
    assert agent.capability_validation_status == "CLOSURE_DECLINED"
    assert git.commands == []
    assert len(writer.calls) == 3


def test_final_approval_without_validated_change_runs_no_git(tmp_path, monkeypatch) -> None:
    agent, _ = _agent(tmp_path, _Writer()), None
    git = _Git()
    monkeypatch.setattr(agent, "_git_command", git.run)

    assert "No hay una mejora VALIDATED" in agent.close_validated_capability_plan("unit.temperature-conversion", approved=True)
    assert git.commands == []


def test_final_closure_aborts_when_validated_file_changes(tmp_path, monkeypatch) -> None:
    agent, _ = _validated_agent(tmp_path, monkeypatch)
    git = _Git()
    monkeypatch.setattr(agent, "_git_command", git.run)
    (tmp_path / _ALLOWED[0]).write_text("changed later", encoding="utf-8")

    response = agent.close_validated_capability_plan("unit.temperature-conversion", approved=True)

    assert "cambió desde la validación" in response
    assert agent.capability_validation_status == "VALIDATED"
    assert git.commands == []


def test_final_closure_aborts_when_staged_scope_is_not_exact(tmp_path, monkeypatch) -> None:
    agent, _ = _validated_agent(tmp_path, monkeypatch)
    git = _Git(staged_names=(_ALLOWED[0], "unrelated.txt"))
    monkeypatch.setattr(agent, "_git_command", git.run)

    response = agent.close_validated_capability_plan("unit.temperature-conversion", approved=True)

    assert "preflight no coincide exactamente" in response
    assert agent.capability_validation_status == "VALIDATED"
    assert [command[0] for command in git.commands] == ["status"]


def test_final_commit_failure_keeps_validated_state(tmp_path, monkeypatch) -> None:
    agent, _ = _validated_agent(tmp_path, monkeypatch)
    git = _Git(commit_returncode=1)
    monkeypatch.setattr(agent, "_git_command", git.run)

    response = agent.close_validated_capability_plan("unit.temperature-conversion", approved=True)

    assert "commit aprobado falló" in response
    assert agent.capability_validation_status == "VALIDATED"
    assert [command[0] for command in git.commands] == ["status", "commit"]