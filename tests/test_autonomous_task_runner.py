from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from agents.coding_agent import CodingAgent, PendingCodingChangeError
from core.autonomous_task_runner import (
    AutonomousFileChange,
    AutonomousIterationRecord,
    AutonomousPlan,
    AutonomousRunnerStatus,
    AutonomousTaskConfig,
    AutonomousTaskRunner,
    MAX_AUTONOMOUS_ITERATIONS,
    ModelPlanner,
    REPLACE_TEXT_OPERATION,
    TaskGoalVerifier,
    WORKER_SYSTEM_PROMPT,
    _history_prompt_lines,
)
from core.git_checkpoint import GitCheckpointManager
from core.goal_evaluator import GoalEvaluator
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


class RecordingReviewer:
    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = list(verdicts)
        self.evaluations: list[object] = []

    def __call__(self, evaluation: object) -> bool:
        self.evaluations.append(evaluation)
        return self._verdicts.pop(0)


def _config(tmp_path: Path, *, max_iterations: int = 3) -> AutonomousTaskConfig:
    return AutonomousTaskConfig(
        goal="hacer que los tests pasen",
        allowed_paths=("pkg",),
        test_paths=("tests/test_autonomous_fake.py",),
        max_iterations=max_iterations,
    )


def _plan(path: str, content: str) -> AutonomousPlan:
    return AutonomousPlan(
        reasoning="cambio mínimo", changes=(AutonomousFileChange(path, content),)
    )


def _runner(
    tmp_path: Path,
    planner: ScriptedPlanner,
    tests: FakeTestRunner,
    *,
    reviewer: RecordingReviewer | None = None,
    max_iterations: int = 3,
) -> AutonomousTaskRunner:
    config = _config(tmp_path, max_iterations=max_iterations)
    return AutonomousTaskRunner(
        tmp_path,
        config,
        planner=planner,
        test_runner=tests,
        evaluator=GoalEvaluator(TaskGoalVerifier()),
        reviewer=reviewer,
        checkpoint_manager=GitCheckpointManager(tmp_path, allowed_scope=("pkg",)),
    )


# A. success en primera iteración


def test_success_in_first_iteration_writes_change_and_accepts(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = ScriptedPlanner([_plan("pkg/module.py", "fixed\n")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert result.reason == "verified"
    assert len(result.iterations) == 1
    assert result.iterations[0].outcome == "SUCCESS"
    assert result.iterations[0].test_passed is True
    assert (tmp_path / "pkg" / "module.py").read_text(encoding="utf-8") == "fixed\n"
    assert result.checkpoint_events == ("checkpoint_created", "write", "accepted")


# B. test fail → retry → success


def test_test_failure_retries_then_succeeds(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = ScriptedPlanner(
        [_plan("pkg/module.py", "broken\n"), _plan("pkg/module.py", "fixed\n")]
    )
    tests = FakeTestRunner([False, True])

    result = _runner(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert len(result.iterations) == 2
    assert result.iterations[0].outcome == "RETRY"
    assert result.iterations[0].restored is True
    assert result.iterations[1].outcome == "SUCCESS"
    assert (tmp_path / "pkg" / "module.py").read_text(encoding="utf-8") == "fixed\n"


# C. límite de iteraciones → BLOCKED


def test_iteration_limit_blocks_with_reason(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    planner = ScriptedPlanner(
        [
            _plan("pkg/one.py", "a\n"),
            _plan("pkg/two.py", "b\n"),
        ]
    )
    tests = FakeTestRunner([False, False])

    result = _runner(tmp_path, planner, tests, max_iterations=2).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "max_iterations_reached"
    assert len(result.iterations) == 2
    assert all(record.restored for record in result.iterations)


def test_max_iterations_is_hard_capped() -> None:
    with pytest.raises(ValueError, match="between 1 and"):
        AutonomousTaskConfig(
            goal="meta",
            allowed_paths=("pkg",),
            test_paths=("tests/test_x.py",),
            max_iterations=MAX_AUTONOMOUS_ITERATIONS + 1,
        )


# D. modificación fuera de allowed_paths → BLOCKED


@pytest.mark.parametrize("path", ["other/module.py", "../outside.py", ".env"])
def test_out_of_scope_modification_blocks_before_writing(
    tmp_path: Path, path: str
) -> None:
    (tmp_path / "pkg").mkdir()
    planner = ScriptedPlanner([_plan(path, "malicious\n")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason.startswith("out_of_scope:")
    assert not list(tmp_path.rglob("*malicious*"))
    assert (tmp_path / ".env").exists() is False


def test_out_of_scope_change_never_creates_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    planner = ScriptedPlanner([_plan("other/module.py", "x\n")])

    result = _runner(tmp_path, planner, FakeTestRunner([True])).run()

    assert "checkpoint_created" not in result.checkpoint_events


# E. evaluator inconclusive → reviewer → success


def test_inconclusive_with_approving_reviewer_and_passing_tests_succeeds(
    tmp_path: Path,
) -> None:
    planner = ScriptedPlanner([AutonomousPlan(reasoning="no hay cambios que hacer")])
    tests = FakeTestRunner([True])
    reviewer = RecordingReviewer([True])

    result = _runner(tmp_path, planner, tests, reviewer=reviewer).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert result.reason == "reviewer_confirmed"
    assert result.iterations[0].reviewer_consulted is True
    assert result.iterations[0].reviewer_approved is True
    assert len(reviewer.evaluations) == 1


# F. reviewer no resuelve → BLOCKED


def test_reviewer_rejection_blocks(tmp_path: Path) -> None:
    planner = ScriptedPlanner([AutonomousPlan(reasoning="sin cambios")])
    reviewer = RecordingReviewer([False])

    result = _runner(
        tmp_path, planner, FakeTestRunner([True]), reviewer=reviewer
    ).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "reviewer_rejected"
    assert result.iterations[0].reviewer_approved is False


def test_reviewer_error_blocks_with_reason(tmp_path: Path) -> None:
    planner = ScriptedPlanner([AutonomousPlan(reasoning="sin cambios")])

    class ExplodingReviewer:
        def __call__(self, evaluation: object) -> bool:
            raise RuntimeError("reviewer offline")

    result = _runner(
        tmp_path, planner, FakeTestRunner([True]), reviewer=ExplodingReviewer()
    ).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "reviewer_error:RuntimeError"


def test_inconclusive_without_reviewer_blocks(tmp_path: Path) -> None:
    planner = ScriptedPlanner([AutonomousPlan(reasoning="sin cambios")])

    result = _runner(tmp_path, planner, FakeTestRunner([True])).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "blocked:INSUFFICIENT_EVIDENCE"


# G. rollback/restauración tras fallo


def test_rollback_restores_original_content_after_failure(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = ScriptedPlanner([_plan("pkg/module.py", "broken\n")])
    tests = FakeTestRunner([False])

    result = _runner(tmp_path, planner, tests, max_iterations=1).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "max_iterations_reached"
    assert (tmp_path / "pkg" / "module.py").read_text(encoding="utf-8") == "original\n"
    assert "restored" in result.checkpoint_events


def test_no_progress_repeated_plan_blocks(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    repeated = _plan("pkg/module.py", "same\n")
    planner = ScriptedPlanner([repeated, repeated])

    result = _runner(tmp_path, planner, FakeTestRunner([False, False])).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "no_progress_repeated_plan"


# H. nunca push/reset/stash


def test_runner_never_spawns_subprocess_git_or_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess execution is forbidden in the runner")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "call", _forbidden)
    monkeypatch.setattr(subprocess, "check_output", _forbidden)
    (tmp_path / "pkg").mkdir()
    planner = ScriptedPlanner(
        [_plan("pkg/module.py", "broken\n"), _plan("pkg/module.py", "fixed\n")]
    )

    result = _runner(tmp_path, planner, FakeTestRunner([False, True])).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS


# I. modo humano existente de CodingAgent sigue funcionando


class _Client:
    def ask(self, *, model: str, messages: list[dict[str, str]]) -> str:
        return "new content\n"


class _Reader:
    def execute(self, path: str) -> str:
        return "old content\n"


class _Writer:
    def execute(self, path: str, content: str) -> str:
        return "written"


def _agent() -> CodingAgent:
    return CodingAgent(_Client(), _Reader(), _Writer())  # type: ignore[arg-type]


def _pending_token(agent: CodingAgent) -> str:
    agent.run("model", [{"role": "user", "content": "corrige agents/coding_agent.py"}])
    pending = agent._pending_change
    assert pending is not None
    return pending.token


def test_delegated_mode_is_opt_in_and_bounded(tmp_path: Path) -> None:
    agent = _agent()
    with pytest.raises(PendingCodingChangeError):
        agent.delegated_authorize_pending_change()

    agent.enable_delegated_authorization(
        allowed_paths=("agents",), max_iterations=1
    )
    token = _pending_token(agent)
    change = agent.delegated_authorize_pending_change()
    assert change.relative_path == "agents/coding_agent.py"
    assert change.token == token
    with pytest.raises(PendingCodingChangeError, match="presupuesto"):
        agent.run("model", [{"role": "user", "content": "corrige agents/coding_agent.py"}])
        agent.delegated_authorize_pending_change()


def test_delegated_mode_rejects_out_of_scope_and_secrets() -> None:
    agent = _agent()
    with pytest.raises(PendingCodingChangeError, match="secretos"):
        agent.enable_delegated_authorization(
            allowed_paths=(".env",), max_iterations=1
        )
    agent.enable_delegated_authorization(allowed_paths=("docs",), max_iterations=2)
    agent.run("model", [{"role": "user", "content": "corrige agents/coding_agent.py"}])
    with pytest.raises(PendingCodingChangeError, match="fuera del alcance delegado"):
        agent.delegated_authorize_pending_change()
    assert agent._pending_change is None


def test_human_token_flow_still_works_alongside_delegation() -> None:
    agent = _agent()
    agent.enable_delegated_authorization(allowed_paths=("docs",), max_iterations=2)
    response = agent.run(
        "model", [{"role": "user", "content": "corrige agents/coding_agent.py"}]
    )
    token = re.search(r"APLICAR ([A-Za-z0-9_-]+)", response).group(1)  # type: ignore[union-attr]
    change = agent.authorize_pending_change(token)
    assert change.relative_path == "agents/coding_agent.py"
    with pytest.raises(PendingCodingChangeError):
        agent.authorize_pending_change(token)


def test_delegation_can_be_disabled() -> None:
    agent = _agent()
    agent.enable_delegated_authorization(allowed_paths=("docs",), max_iterations=1)
    assert agent.delegated_authorization_active is True
    agent.disable_delegated_authorization()
    assert agent.delegated_authorization_active is False
    with pytest.raises(PendingCodingChangeError, match="no está activado"):
        agent.delegated_authorize_pending_change()


# Config validation


def test_config_requires_paths_and_bounds() -> None:
    with pytest.raises(ValueError):
        AutonomousTaskConfig(
            goal="meta", allowed_paths=(), test_paths=("tests/x.py",), max_iterations=1
        )
    with pytest.raises(ValueError):
        AutonomousTaskConfig(
            goal="meta", allowed_paths=("pkg",), test_paths=(), max_iterations=1
        )
    with pytest.raises(ValueError, match="secretos|secret"):
        AutonomousTaskConfig(
            goal="meta",
            allowed_paths=("config/.env",),
            test_paths=("tests/x.py",),
            max_iterations=1,
        )


# ModelPlanner JSON parsing with a fake provider


class FakeProvider:
    def __init__(self, content: object) -> None:
        self._content = content

    def chat(self, *, model: str, messages: list[dict[str, str]], stream: bool) -> dict:
        return {"model": model, "message": {"content": self._content}}


def test_model_planner_parses_strict_json_payload() -> None:
    planner = ModelPlanner(
        FakeProvider(
            '{"reasoning": "arreglo el bug", '
            '"changes": [{"path": "pkg/module.py", "content": "x\\n"}]}'
        ),
        "glm-5.3-flash",
    )

    plan = planner("meta", 1, ())

    assert plan.reasoning == "arreglo el bug"
    assert plan.changes == (AutonomousFileChange("pkg/module.py", "x\n"),)


def test_model_planner_rejects_malformed_payload() -> None:
    planner = ModelPlanner(FakeProvider("no soy json"), "glm-5.3-flash")

    with pytest.raises(ValueError):
        planner("meta", 1, ())


# J. feedback del intento anterior hacia el planner


class FeedbackTestRunner:
    def __init__(self, results: list[TestRunResult]) -> None:
        self._results = list(results)

    def run(self, test_paths: tuple[str, ...]) -> TestRunResult:
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


class CapturingProvider:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def chat(self, *, model: str, messages: list[dict[str, str]], stream: bool) -> dict:
        self.prompts.append(messages[-1]["content"])
        return {"model": model, "message": {"content": self._payloads.pop(0)}}


def _failed_result(output_tail: str, *, timed_out: bool = False) -> TestRunResult:
    return TestRunResult(
        passed=False,
        exit_code=None if timed_out else 1,
        timed_out=timed_out,
        detail=(
            "pytest timed out after 120.0 seconds."
            if timed_out
            else "pytest exited with code 1."
        ),
        output_tail=output_tail,
        command=("pytest", "tests/test_autonomous_fake.py"),
        basetemp=None,
    )


def _passed_result() -> TestRunResult:
    return TestRunResult(
        passed=True,
        exit_code=0,
        timed_out=False,
        detail="pytest exited with code 0.",
        output_tail="1 passed in 0.01s",
        command=("pytest", "tests/test_autonomous_fake.py"),
        basetemp=None,
    )


class RecordingHistoryPlanner:
    def __init__(self, plans: list[AutonomousPlan]) -> None:
        self._plans = list(plans)
        self.histories: list[tuple[AutonomousIterationRecord, ...]] = []

    def __call__(
        self, goal: str, iteration: int, history: object
    ) -> AutonomousPlan:
        self.histories.append(tuple(history))  # type: ignore[arg-type]
        return self._plans.pop(0)


def _runner_with_history(
    tmp_path: Path, planner: object, tests: FeedbackTestRunner
) -> AutonomousTaskRunner:
    return AutonomousTaskRunner(
        tmp_path,
        _config(tmp_path, max_iterations=2),
        planner=planner,  # type: ignore[arg-type]
        test_runner=tests,
        evaluator=GoalEvaluator(TaskGoalVerifier()),
        checkpoint_manager=GitCheckpointManager(tmp_path, allowed_scope=("pkg",)),
    )


# A + B. fallo en iteración 1 → rollback → iteración 2 recibe el feedback


def test_failed_iteration_feedback_reaches_planner_history(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = RecordingHistoryPlanner(
        [_plan("pkg/module.py", "broken\n"), _plan("pkg/module.py", "fixed\n")]
    )
    tests = FeedbackTestRunner(
        [
            _failed_result(
                "AssertionError: expected 5 got 4\nFAILED tests/test_autonomous_fake.py"
            ),
            _passed_result(),
        ]
    )

    result = _runner_with_history(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    # A. iteración 1: fallo y rollback
    assert result.iterations[0].outcome == "RETRY"
    assert result.iterations[0].restored is True
    assert "restored" in result.checkpoint_events
    # B. iteración 2 recibe el feedback del intento anterior
    second_history = planner.histories[1]
    assert len(second_history) == 1
    record = second_history[0]
    assert record.test_exit_code == 1
    assert record.test_timed_out is False
    assert "expected 5 got 4" in record.test_output_tail
    assert record.evaluation_status == "RETRY"
    assert record.evaluation_reason == "tests_failed"
    assert record.changed_paths == ("pkg/module.py",)


def test_model_planner_prompt_includes_previous_failure_feedback(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    provider = CapturingProvider(
        [
            '{"reasoning": "intento 1", "changes": '
            '[{"path": "pkg/module.py", "content": "broken\\n"}]}',
            '{"reasoning": "intento 2", "changes": '
            '[{"path": "pkg/module.py", "content": "fixed\\n"}]}',
        ]
    )
    planner = ModelPlanner(provider, "glm-5.3-flash")
    tests = FeedbackTestRunner(
        [_failed_result("AssertionError: expected 5 got 4"), _passed_result()]
    )

    result = _runner_with_history(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert len(provider.prompts) == 2
    prompt = provider.prompts[1]
    assert "exit_code: 1" in prompt
    assert "expected 5 got 4" in prompt
    assert "tests_failed" in prompt
    assert "changed_paths: pkg/module.py" in prompt


# C. tests PASS → no se inventa feedback de error


def test_passing_history_does_not_invent_failure_feedback() -> None:
    record = AutonomousIterationRecord(
        iteration=1,
        outcome="SUCCESS",
        reasoning="ok",
        changed_paths=("pkg/module.py",),
        evaluation_status="SUCCESS",
        evaluation_reason="verified",
        test_passed=True,
        test_exit_code=0,
        test_timed_out=False,
        test_output_tail="1 passed in 0.01s",
    )

    prompt = "\n".join(_history_prompt_lines((record,)))

    assert "exit_code" not in prompt
    assert "timed_out" not in prompt
    assert "Salida de tests" not in prompt
    assert "1 passed" not in prompt


def test_test_output_tail_is_truncated_in_planner_prompt() -> None:
    tail = "x" * 2000 + "REAL_ERROR"
    record = AutonomousIterationRecord(
        iteration=1,
        outcome="RETRY",
        reasoning="intento",
        changed_paths=("pkg/module.py",),
        evaluation_status="RETRY",
        evaluation_reason="tests_failed",
        test_passed=False,
        test_exit_code=1,
        test_timed_out=False,
        test_output_tail=tail,
    )

    prompt = "\n".join(_history_prompt_lines((record,)))

    assert "REAL_ERROR" in prompt
    assert "x" * 1000 not in prompt


# D. timeout → planner recibe timed_out=True


def test_timeout_feedback_reaches_planner_history(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = RecordingHistoryPlanner(
        [_plan("pkg/module.py", "hang\n"), _plan("pkg/module.py", "fixed\n")]
    )
    tests = FeedbackTestRunner(
        [
            _failed_result("pytest timed out after 120.0 seconds.", timed_out=True),
            _passed_result(),
        ]
    )

    result = _runner_with_history(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    record = planner.histories[1][-1]
    assert record.test_timed_out is True
    assert record.test_exit_code is None
    assert record.test_passed is False


def test_model_planner_prompt_marks_timeout(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    provider = CapturingProvider(
        [
            '{"reasoning": "a", "changes": [{"path": "pkg/module.py", "content": "hang\\n"}]}',
            '{"reasoning": "b", "changes": [{"path": "pkg/module.py", "content": "fixed\\n"}]}',
        ]
    )
    planner = ModelPlanner(provider, "glm-5.3-flash")
    tests = FeedbackTestRunner(
        [
            _failed_result("pytest timed out after 120.0 seconds.", timed_out=True),
            _passed_result(),
        ]
    )

    result = _runner_with_history(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert "timed_out: True" in provider.prompts[1]

# ---------------------------------------------------------------------------
# Cambios acotados: replace_text
# ---------------------------------------------------------------------------


def _replace_plan(path: str, old_text: str, new_text: str) -> AutonomousPlan:
    return AutonomousPlan(
        reasoning="cambio mínimo",
        changes=(
            AutonomousFileChange(
                path,
                operation=REPLACE_TEXT_OPERATION,
                old_text=old_text,
                new_text=new_text,
            ),
        ),
    )


# A. replace_text correcto modifica solo el fragmento esperado


def test_replace_text_changes_only_expected_fragment(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text(
        "VALUE = 1\nOTHER = 9\n", encoding="utf-8"
    )
    planner = ScriptedPlanner([_replace_plan("pkg/module.py", "VALUE = 1", "VALUE = 2")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert (
        tmp_path / "pkg" / "module.py"
    ).read_text(encoding="utf-8") == "VALUE = 2\nOTHER = 9\n"


# B. old_text inexistente → fallo del cambio sin corrupción


def test_replace_text_missing_old_text_fails_without_corruption(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = ScriptedPlanner([_replace_plan("pkg/module.py", "NO_EXISTE", "x")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests, max_iterations=1).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "max_iterations_reached"
    assert result.iterations[0].restored is True
    assert (
        tmp_path / "pkg" / "module.py"
    ).read_text(encoding="utf-8") == "original\n"


# C. old_text ambiguo → fallo sin escribir


def test_replace_text_ambiguous_old_text_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text(
        "VALUE = 1\nVALUE = 1\n", encoding="utf-8"
    )
    planner = ScriptedPlanner([_replace_plan("pkg/module.py", "VALUE = 1", "VALUE = 2")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests, max_iterations=1).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "max_iterations_reached"
    assert result.iterations[0].restored is True
    assert (
        tmp_path / "pkg" / "module.py"
    ).read_text(encoding="utf-8") == "VALUE = 1\nVALUE = 1\n"


# D. replace_text fuera de allowed_paths → BLOCKED antes de escribir


def test_replace_text_out_of_scope_path_blocks(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    planner = ScriptedPlanner([_replace_plan("other/module.py", "VALUE = 1", "VALUE = 2")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason.startswith("out_of_scope:")
    assert (
        tmp_path / "other" / "module.py"
    ).read_text(encoding="utf-8") == "VALUE = 1\n"


# E. rollback restaura el archivo tras replace_text con tests fallidos


def test_replace_text_rollback_restores_original_file(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("original\n", encoding="utf-8")
    planner = ScriptedPlanner([_replace_plan("pkg/module.py", "original", "broken")])
    tests = FakeTestRunner([False])

    result = _runner(tmp_path, planner, tests, max_iterations=1).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "max_iterations_reached"
    assert (
        tmp_path / "pkg" / "module.py"
    ).read_text(encoding="utf-8") == "original\n"
    assert "restored" in result.checkpoint_events


def test_replace_text_requires_exact_unique_old_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AutonomousFileChange(
            "pkg/module.py",
            operation=REPLACE_TEXT_OPERATION,
            old_text="",
            new_text="x",
        )
    with pytest.raises(ValueError):
        AutonomousFileChange(
            "pkg/module.py",
            operation=REPLACE_TEXT_OPERATION,
            old_text="a",
            new_text="",
        )
    with pytest.raises(ValueError):
        AutonomousFileChange(
            "pkg/module.py",
            operation="free_edit",
            old_text="a",
            new_text="b",
        )


# F. el formato full-file actual sigue funcionando


def test_full_file_format_still_works_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "new.py").write_text("old\n", encoding="utf-8")
    planner = ScriptedPlanner([_plan("pkg/new.py", "complete file\n")])
    tests = FakeTestRunner([True])

    result = _runner(tmp_path, planner, tests).run()

    assert result.status is AutonomousRunnerStatus.SUCCESS
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "complete file\n"


def test_model_planner_parses_replace_text_and_legacy_payload() -> None:
    planner = ModelPlanner(
        FakeProvider(
            '{"reasoning": "cambio acotado", "changes": ['
            '{"op": "replace_text", "path": "pkg/module.py", '
            '"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]}'
        ),
        "glm-5.3-flash",
    )

    plan = planner("meta", 1, ())

    assert plan.changes == (
        AutonomousFileChange(
            "pkg/module.py",
            operation=REPLACE_TEXT_OPERATION,
            old_text="VALUE = 1",
            new_text="VALUE = 2",
        ),
    )


def test_model_planner_parses_append_text_payload() -> None:
    planner = ModelPlanner(
        FakeProvider(
            '{"reasoning": "anado helper", "changes": ['
            '{"op": "append_text", "path": "pkg/module.py", "content": "\\n# helper\\n"}]}'
        ),
        "glm-5.3-flash",
    )

    plan = planner("meta", 1, ())

    assert plan.changes[0].operation == "append_text"
    assert plan.changes[0].content == "\n# helper\n"


def test_model_planner_rejects_unknown_operation() -> None:
    planner = ModelPlanner(
        FakeProvider(
            '{"reasoning": "x", "changes": ['
            '{"op": "free_edit", "path": "pkg/module.py"}]}'
        ),
        "glm-5.3-flash",
    )

    with pytest.raises(ValueError):
        planner("meta", 1, ())


# G. el prompt del planner pide cambios mínimos


def test_worker_prompt_requests_minimal_changes() -> None:
    assert "replace_text" in WORKER_SYSTEM_PROMPT
    assert "append_text" in WORKER_SYSTEM_PROMPT
    assert "full_file" in WORKER_SYSTEM_PROMPT
    assert "NUNCA devuelvas el archivo completo" in WORKER_SYSTEM_PROMPT
    assert "old_text" in WORKER_SYSTEM_PROMPT
    assert "ÚNICO" in WORKER_SYSTEM_PROMPT
