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