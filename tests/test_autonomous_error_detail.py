"""Sanitization and propagation of real inference error details into reasons."""

from __future__ import annotations

from pathlib import Path

from core.autonomous_task_runner import (
    AutonomousPlan,
    AutonomousRunnerStatus,
    AutonomousTaskConfig,
    AutonomousTaskRunner,
    MAX_ERROR_DETAIL_CHARS,
    TaskGoalVerifier,
    _error_detail,
    _failure_reason,
    _sanitize_error_detail,
)
from core.git_checkpoint import GitCheckpointManager
from core.goal_evaluator import GoalEvaluator
from models.chat_inference import ChatInferenceError

from tests.test_autonomous_task_runner import FakeTestRunner

import pytest


class ScriptedPlanner:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __call__(self, goal: str, iteration: int, history: object) -> AutonomousPlan:
        raise self._error


def _runner(tmp_path: Path, planner: ScriptedPlanner) -> AutonomousTaskRunner:
    (tmp_path / "pkg").mkdir(exist_ok=True)
    return AutonomousTaskRunner(
        tmp_path,
        AutonomousTaskConfig(
            goal="objetivo",
            allowed_paths=("pkg",),
            test_paths=("tests/test_fake.py",),
            max_iterations=1,
        ),
        planner=planner,
        test_runner=FakeTestRunner([True]),
        evaluator=GoalEvaluator(TaskGoalVerifier()),
        checkpoint_manager=GitCheckpointManager(tmp_path, allowed_scope=("pkg",)),
    )


# A. ChatInferenceError con mensaje "429 RESOURCE_EXHAUSTED"


def test_chat_inference_error_message_is_propagated(tmp_path: Path) -> None:
    error = ChatInferenceError("gemini", "gemini-model", "429 RESOURCE_EXHAUSTED")

    result = _runner(tmp_path, ScriptedPlanner(error)).run()

    assert result.status is AutonomousRunnerStatus.BLOCKED
    assert result.reason == "planner_error:ChatInferenceError:429 RESOURCE_EXHAUSTED"
    assert "429 RESOURCE_EXHAUSTED" in result.reason


def test_error_detail_includes_plain_message() -> None:
    error = ChatInferenceError("gemini", "gemini-model", "429 RESOURCE_EXHAUSTED")

    assert _error_detail(error) == "429 RESOURCE_EXHAUSTED"


# B. Excepción encadenada con ReadTimeout


def test_chained_read_timeout_cause_is_propagated() -> None:
    cause = TimeoutError("ReadTimeout: read timed out")
    error = ChatInferenceError("gemini", "gemini-model", "gemini chat failed")
    error.__cause__ = cause

    detail = _error_detail(error)

    assert "ReadTimeout" in detail
    assert "TimeoutError: ReadTimeout: read timed out" in detail


# C. Sanitizado de secretos


@pytest.mark.parametrize(
    "text",
    (
        "Authorization: Bearer SECRET",
        "request failed with Bearer abc123token",
        "api_key=AIzaSuperSecretValue",
        "url https://host/v1?key=AIzaSuperSecretValue&alt=sse",
        "token: s3cr3t-value password=hunter2",
    ),
)
def test_secrets_are_redacted(text: str) -> None:
    sanitized = _sanitize_error_detail(text)

    assert "SECRET" not in sanitized
    assert "abc123token" not in sanitized
    assert "AIzaSuperSecretValue" not in sanitized
    assert "s3cr3t-value" not in sanitized
    assert "hunter2" not in sanitized
    assert "[REDACTED]" in sanitized


def test_reason_with_authorization_header_hides_secret() -> None:
    error = ChatInferenceError(
        "gemini", "gemini-model", "Authorization: Bearer SECRET rejected"
    )

    reason = _failure_reason("planner_error", error)

    assert reason.startswith("planner_error:ChatInferenceError:")
    assert "SECRET" not in reason
    assert "[REDACTED]" in reason


# D. Mensaje > 500 chars → truncado


def test_long_message_is_truncated_to_limit() -> None:
    error = ChatInferenceError("gemini", "gemini-model", "x" * 5000)

    detail = _error_detail(error)

    assert len(detail) == MAX_ERROR_DETAIL_CHARS


def test_long_reason_is_truncated_and_keeps_prefix() -> None:
    error = ChatInferenceError("gemini", "gemini-model", "y" * 5000)

    reason = _failure_reason("reviewer_error", error)

    assert reason.startswith("reviewer_error:ChatInferenceError:")
    assert len(reason) < 600


# 5. Sin detalle útil → se conserva el reason actual


def test_empty_message_keeps_legacy_reason() -> None:
    error = ChatInferenceError("gemini", "gemini-model", "")

    assert _failure_reason("planner_error", error) == "planner_error:ChatInferenceError"


def test_status_code_is_included_when_available() -> None:
    class ProviderError(Exception):
        status_code = 429

    error = ProviderError("quota exceeded")
    error.__cause__ = None

    detail = _error_detail(error)

    assert detail == "quota exceeded (status_code=429)"
