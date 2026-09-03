from __future__ import annotations

from pathlib import Path

import pytest

from core.test_runner import PytestRunner


PASSING_TEST = "def test_ok():\n    assert True\n"
FAILING_TEST = "def test_fail():\n    assert False\n"
SLOW_TEST = "import time\n\ndef test_slow():\n    time.sleep(30)\n"


def _runner(tmp_path: Path, **kwargs) -> PytestRunner:
    return PytestRunner(tmp_path, **kwargs)


def test_passing_single_path_returns_structured_success(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text(PASSING_TEST, encoding="utf-8")

    result = _runner(tmp_path).run(("test_ok.py",), timeout=60)

    assert result.passed
    assert result.exit_code == 0
    assert not result.timed_out
    assert "1 passed" in result.output_tail
    assert "--basetemp" in " ".join(result.command)
    assert result.basetemp is not None
    assert not result.basetemp.exists()


def test_failing_path_reports_failure_and_diagnostic_output(tmp_path: Path) -> None:
    (tmp_path / "test_fail.py").write_text(FAILING_TEST, encoding="utf-8")

    result = _runner(tmp_path).run(("test_fail.py",), timeout=60)

    assert not result.passed
    assert result.exit_code == 1
    assert not result.timed_out
    assert "1 failed" in result.output_tail
    assert "assert False" in result.output_tail


def test_multiple_paths_are_parametrizable(tmp_path: Path) -> None:
    (tmp_path / "test_one.py").write_text(PASSING_TEST, encoding="utf-8")
    (tmp_path / "test_two.py").write_text(PASSING_TEST, encoding="utf-8")

    result = _runner(tmp_path).run(("test_one.py", "test_two.py"), timeout=60)

    assert result.passed
    assert "2 passed" in result.output_tail


def test_timeout_is_configurable_and_reported(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text(SLOW_TEST, encoding="utf-8")

    result = _runner(tmp_path).run(("test_slow.py",), timeout=2)

    assert not result.passed
    assert result.timed_out
    assert result.exit_code is None
    assert "timed out" in result.detail


def test_explicit_basetemp_is_used_and_cleanup_is_optional(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text(PASSING_TEST, encoding="utf-8")
    basetemp = tmp_path / "isolated-basetemp"
    runner = _runner(tmp_path)

    kept = runner.run(("test_ok.py",), timeout=60, basetemp=basetemp, cleanup_basetemp=False)
    assert kept.passed
    assert kept.basetemp == basetemp
    assert basetemp.exists()

    cleaned = runner.run(("test_ok.py",), timeout=60, basetemp=basetemp, cleanup_basetemp=True)
    assert cleaned.passed
    assert not basetemp.exists()


def test_paths_outside_project_root_are_rejected(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    with pytest.raises(ValueError, match="project root"):
        runner.run(("../escaped_test.py",))
    with pytest.raises(ValueError):
        runner.run(())
