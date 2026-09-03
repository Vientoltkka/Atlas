"""Minimal bounded autonomous task runner.

Composes the existing 1A primitives (GoalEvaluator, PytestRunner,
GitCheckpointManager) into a single GOAL -> PLAN -> MODIFY -> TEST ->
EVALUATE loop that always terminates in SUCCESS or BLOCKED.

Safety invariants, by construction:

- No git process is ever spawned by this module: push, reset and stash are
  structurally impossible. Snapshots and restores go exclusively through
  GitCheckpointManager, which refuses secret files and out-of-scope paths.
- ``max_iterations`` is mandatory and hard-capped (MAX_AUTONOMOUS_ITERATIONS).
- ``allowed_paths`` and ``test_paths`` are mandatory; any modification outside
  the sanctioned scope blocks the task before a single write happens.
- Every iteration produces an auditable structured record.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

from core.git_checkpoint import GitCheckpointManager
from core.goal_evaluator import GoalEvaluation, GoalEvaluationStatus, GoalEvaluator
from core.goal_verifier import (
    GoalVerificationReason,
    GoalVerificationResult,
    GoalVerificationStatus,
)
from core.model_manager import ModelDescriptor, ModelManager, ModelSelectionRequest
from core.test_runner import PytestRunner, TestRunResult

MAX_AUTONOMOUS_ITERATIONS = 20
MAX_PLAN_CHANGES = 16
MAX_HISTORY_ITEMS_IN_PROMPT = 5
MAX_TEST_OUTPUT_CHARS_IN_PROMPT = 800
WORKER_ROLE_ENV = "ATLAS_AUTONOMY_WORKER_MODEL"
REVIEWER_ROLE_ENV = "ATLAS_AUTONOMY_REVIEWER_MODEL"
LOCAL_ROLE_ENV = "ATLAS_AUTONOMY_LOCAL_MODEL"
DEFAULT_REVIEWER_LOGICAL_ID = "chat-gemini"
DEFAULT_LOCAL_LOGICAL_ID = "chat-local"


class AutonomousRunnerError(RuntimeError):
    """Base error for the autonomous task runner contracts."""


class AutonomousRunnerStatus(str, Enum):
    """Terminal states of one bounded autonomous task."""

    SUCCESS = "SUCCESS"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class AutonomousFileChange:
    """One whole-file replacement scoped to the allowed paths."""

    relative_path: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path.strip():
            raise ValueError("relative_path must be a non-empty string.")
        if not isinstance(self.content, str):
            raise ValueError("content must be text.")


@dataclass(frozen=True, slots=True)
class AutonomousPlan:
    """Worker output for one iteration: reasoning plus bounded file changes."""

    reasoning: str
    changes: tuple[AutonomousFileChange, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string.")
        object.__setattr__(self, "changes", tuple(self.changes))
        if len(self.changes) > MAX_PLAN_CHANGES:
            raise ValueError("plan exceeds the safe change limit.")
        if not all(isinstance(change, AutonomousFileChange) for change in self.changes):
            raise ValueError("changes must contain AutonomousFileChange values.")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [[change.relative_path, change.content] for change in self.changes],
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AutonomousTaskConfig:
    """Mandatory bounded configuration for one autonomous task."""

    goal: str
    allowed_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    max_iterations: int

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-empty string.")
        object.__setattr__(self, "allowed_paths", _normalized_paths(self.allowed_paths, "allowed_paths"))
        object.__setattr__(self, "test_paths", _normalized_paths(self.test_paths, "test_paths"))
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise ValueError("max_iterations must be an integer.")
        if self.max_iterations < 1 or self.max_iterations > MAX_AUTONOMOUS_ITERATIONS:
            raise ValueError(
                f"max_iterations must be between 1 and {MAX_AUTONOMOUS_ITERATIONS}."
            )


@dataclass(frozen=True, slots=True)
class AutonomousModificationResult:
    """Structured evidence of one modify step."""

    applied: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AutonomousIterationRecord:
    """Auditable outcome of exactly one autonomous iteration."""

    iteration: int
    outcome: str
    reasoning: str
    changed_paths: tuple[str, ...]
    evaluation_status: str
    evaluation_reason: str
    test_passed: bool | None = None
    test_exit_code: int | None = None
    test_timed_out: bool = False
    test_output_tail: str = ""
    reviewer_consulted: bool = False
    reviewer_approved: bool | None = None
    restored: bool = False


@dataclass(frozen=True, slots=True)
class AutonomousTaskResult:
    """Final structured result of one bounded autonomous task."""

    status: AutonomousRunnerStatus
    reason: str
    iterations: tuple[AutonomousIterationRecord, ...] = ()
    last_test_result: TestRunResult | None = None
    rolled_back: bool = False
    checkpoint_events: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.status is AutonomousRunnerStatus.SUCCESS


class TaskGoalVerifier:
    """Deterministic integrity verifier that feeds the existing GoalEvaluator.

    It reuses ``GoalVerificationResult`` so ``GoalEvaluator`` keeps full
    control of the SUCCESS / RETRY / BLOCKED mapping. The verifier only checks
    modification integrity; goal satisfaction is asserted by the tests and,
    when evidence is inconclusive, by the injected reviewer.
    """

    def verify(self, plan: object, execution_result: object) -> GoalVerificationResult:
        if not isinstance(plan, AutonomousPlan) or not isinstance(
            execution_result, AutonomousModificationResult
        ):
            return GoalVerificationResult(
                satisfied=False,
                reason=GoalVerificationReason.INVALID_PLAN,
                message="El plan o la evidencia de modificación son inválidos.",
            )
        if execution_result.failed:
            return GoalVerificationResult(
                satisfied=False,
                reason=GoalVerificationReason.PLAN_FAILED,
                evidence=tuple(f"{path}: {error}" for path, error in execution_result.failed),
                message="No se pudieron aplicar todos los cambios planificados.",
            )
        if not plan.changes:
            return GoalVerificationResult(
                satisfied=False,
                reason=GoalVerificationReason.INSUFFICIENT_EVIDENCE,
                verification_status=GoalVerificationStatus.INCONCLUSIVE,
                message=(
                    "El plan no produjo cambios y no existe evidencia suficiente "
                    "para verificar el objetivo."
                ),
            )
        return GoalVerificationResult(
            satisfied=True,
            reason=GoalVerificationReason.SUCCESS,
            evidence=tuple(f"applied:{path}" for path in execution_result.applied),
            message="Los cambios planificados se aplicaron íntegramente.",
        )


class AutonomousPlanner(Protocol):
    def __call__(
        self, goal: str, iteration: int, history: Sequence[AutonomousIterationRecord]
    ) -> AutonomousPlan: ...


class AutonomousReviewer(Protocol):
    def __call__(self, evaluation: GoalEvaluation) -> bool: ...


class AutonomousTaskRunner:
    """Run the bounded autonomous cycle with the 1A primitives."""

    def __init__(
        self,
        project_root: Path,
        config: AutonomousTaskConfig,
        *,
        planner: AutonomousPlanner,
        test_runner: PytestRunner | None = None,
        evaluator: GoalEvaluator | None = None,
        reviewer: AutonomousReviewer | None = None,
        checkpoint_manager: GitCheckpointManager | None = None,
    ) -> None:
        if not isinstance(config, AutonomousTaskConfig):
            raise AutonomousRunnerError("config must be AutonomousTaskConfig.")
        if not callable(planner):
            raise AutonomousRunnerError("planner must be callable.")
        if reviewer is not None and not callable(reviewer):
            raise AutonomousRunnerError("reviewer must be callable or None.")
        self._root = project_root.resolve()
        self._config = config
        self._planner = planner
        self._reviewer = reviewer
        self._tests = test_runner or PytestRunner(self._root)
        self._evaluator = evaluator or GoalEvaluator(TaskGoalVerifier())
        self._checkpoints = checkpoint_manager or GitCheckpointManager(
            self._root, allowed_scope=config.allowed_paths
        )
        self._allowed_dirs = tuple(
            self._validated_scope_entry(entry) for entry in config.allowed_paths
        )
        for entry in config.test_paths:
            self._validated_scope_entry(entry)

    def run(self) -> AutonomousTaskResult:
        history: list[AutonomousIterationRecord] = []
        seen_plan_digests: set[str] = set()
        last_test: TestRunResult | None = None

        for iteration in range(1, self._config.max_iterations + 1):
            try:
                plan = self._planner(self._config.goal, iteration, tuple(history))
            except Exception as error:
                return self._terminal(
                    history,
                    AutonomousRunnerStatus.BLOCKED,
                    f"planner_error:{type(error).__name__}",
                    last_test,
                )
            for change in plan.changes:
                try:
                    self._validated_relative_path(change.relative_path)
                except ValueError:
                    return self._terminal(
                        history,
                        AutonomousRunnerStatus.BLOCKED,
                        f"out_of_scope:{_display_path(change.relative_path)}",
                        last_test,
                    )
            if plan.digest in seen_plan_digests:
                return self._terminal(
                    history,
                    AutonomousRunnerStatus.BLOCKED,
                    "no_progress_repeated_plan",
                    last_test,
                )
            seen_plan_digests.add(plan.digest)

            restored = False
            checkpoint = None
            if plan.changes:
                try:
                    checkpoint = self._checkpoints.checkpoint(
                        [change.relative_path for change in plan.changes]
                    )
                except (RuntimeError, ValueError) as error:
                    return self._terminal(
                        history,
                        AutonomousRunnerStatus.BLOCKED,
                        f"checkpoint_error:{_first_line(str(error))}",
                        last_test,
                    )
                modification = self._apply(plan, checkpoint)
            else:
                modification = AutonomousModificationResult()

            test_result = self._tests.run(self._config.test_paths)
            last_test = test_result
            evaluation = self._evaluator.evaluate(
                plan, modification, test_result=test_result
            )

            if evaluation.status is GoalEvaluationStatus.SUCCESS:
                if checkpoint is not None:
                    self._checkpoints.accept()
                history.append(self._record(
                    iteration, "SUCCESS", plan, modification, test_result, evaluation
                ))
                return self._result(
                    AutonomousRunnerStatus.SUCCESS,
                    evaluation.reason,
                    history,
                    last_test,
                    rolled_back=False,
                )

            if evaluation.status is GoalEvaluationStatus.RETRY:
                if checkpoint is not None:
                    restored = self._safe_restore()
                history.append(self._record(
                    iteration,
                    "RETRY",
                    plan,
                    modification,
                    test_result,
                    evaluation,
                    restored=restored,
                ))
                continue

            if (
                evaluation.verification.verification_status
                is GoalVerificationStatus.INCONCLUSIVE
                and self._reviewer is not None
            ):
                reviewer_approved: bool | None
                try:
                    reviewer_approved = bool(self._reviewer(evaluation))
                except Exception as error:
                    if checkpoint is not None:
                        restored = self._safe_restore()
                    history.append(self._record(
                        iteration,
                        "BLOCKED",
                        plan,
                        modification,
                        test_result,
                        evaluation,
                        reviewer_consulted=True,
                        restored=restored,
                    ))
                    return self._terminal(
                        history,
                        AutonomousRunnerStatus.BLOCKED,
                        f"reviewer_error:{type(error).__name__}",
                        last_test,
                        rolled_back=restored,
                    )
                if reviewer_approved and test_result.passed:
                    if checkpoint is not None:
                        self._checkpoints.accept()
                    history.append(self._record(
                        iteration,
                        "SUCCESS",
                        plan,
                        modification,
                        test_result,
                        evaluation,
                        reviewer_consulted=True,
                        reviewer_approved=True,
                    ))
                    return self._result(
                        AutonomousRunnerStatus.SUCCESS,
                        "reviewer_confirmed",
                        history,
                        last_test,
                        rolled_back=False,
                    )
                if reviewer_approved:
                    if checkpoint is not None:
                        restored = self._safe_restore()
                    history.append(self._record(
                        iteration,
                        "RETRY",
                        plan,
                        modification,
                        test_result,
                        evaluation,
                        reviewer_consulted=True,
                        reviewer_approved=True,
                        restored=restored,
                    ))
                    continue
                if checkpoint is not None:
                    restored = self._safe_restore()
                history.append(self._record(
                    iteration,
                    "BLOCKED",
                    plan,
                    modification,
                    test_result,
                    evaluation,
                    reviewer_consulted=True,
                    reviewer_approved=False,
                    restored=restored,
                ))
                return self._terminal(
                    history,
                    AutonomousRunnerStatus.BLOCKED,
                    "reviewer_rejected",
                    last_test,
                    rolled_back=restored,
                )

            if checkpoint is not None:
                restored = self._safe_restore()
            history.append(self._record(
                iteration,
                "BLOCKED",
                plan,
                modification,
                test_result,
                evaluation,
                restored=restored,
            ))
            return self._terminal(
                history,
                AutonomousRunnerStatus.BLOCKED,
                evaluation.reason,
                last_test,
                rolled_back=restored,
            )

        return self._terminal(
            history, AutonomousRunnerStatus.BLOCKED, "max_iterations_reached", last_test
        )

    def _apply(
        self, plan: AutonomousPlan, checkpoint: object
    ) -> AutonomousModificationResult:
        applied: list[str] = []
        failed: list[tuple[str, str]] = []
        for change in plan.changes:
            try:
                self._checkpoints.write(change.relative_path, change.content)
                applied.append(change.relative_path)
            except (TypeError, ValueError, RuntimeError) as error:
                failed.append((change.relative_path, _first_line(str(error))))
        return AutonomousModificationResult(tuple(applied), tuple(failed))

    def _safe_restore(self) -> bool:
        if self._checkpoints.active is None:
            return False
        try:
            self._checkpoints.restore(reason="autonomous_rollback")
            return True
        except RuntimeError:
            return False

    def _record(
        self,
        iteration: int,
        outcome: str,
        plan: AutonomousPlan,
        modification: AutonomousModificationResult,
        test_result: TestRunResult | None,
        evaluation: GoalEvaluation,
        *,
        reviewer_consulted: bool = False,
        reviewer_approved: bool | None = None,
        restored: bool = False,
    ) -> AutonomousIterationRecord:
        return AutonomousIterationRecord(
            iteration=iteration,
            outcome=outcome,
            reasoning=plan.reasoning,
            changed_paths=tuple(
                dict.fromkeys((*modification.applied, *(path for path, _ in modification.failed)))
            ),
            evaluation_status=evaluation.status.value,
            evaluation_reason=evaluation.reason,
            test_passed=None if test_result is None else test_result.passed,
            test_exit_code=None if test_result is None else test_result.exit_code,
            test_timed_out=False if test_result is None else test_result.timed_out,
            test_output_tail="" if test_result is None else test_result.output_tail,
            reviewer_consulted=reviewer_consulted,
            reviewer_approved=reviewer_approved,
            restored=restored,
        )

    def _terminal(
        self,
        history: list[AutonomousIterationRecord],
        status: AutonomousRunnerStatus,
        reason: str,
        last_test: TestRunResult | None,
        *,
        rolled_back: bool = False,
    ) -> AutonomousTaskResult:
        return self._result(status, reason, history, last_test, rolled_back=rolled_back)

    def _result(
        self,
        status: AutonomousRunnerStatus,
        reason: str,
        history: list[AutonomousIterationRecord],
        last_test: TestRunResult | None,
        *,
        rolled_back: bool,
    ) -> AutonomousTaskResult:
        return AutonomousTaskResult(
            status=status,
            reason=reason,
            iterations=tuple(history),
            last_test_result=last_test,
            rolled_back=rolled_back,
            checkpoint_events=tuple(
                entry["event"] for entry in self._checkpoints.audit_log
            ),
        )

    def _validated_scope_entry(self, entry: str) -> Path:
        relative = _validated_relative(entry)
        resolved = (self._root / relative).resolve()
        if resolved == self._root or self._root not in resolved.parents:
            raise AutonomousRunnerError("scope entries must remain inside the project root.")
        return resolved

    def _validated_relative_path(self, raw: str) -> str:
        relative = _validated_relative(raw)
        resolved = (self._root / relative).resolve()
        if resolved == self._root or self._root not in resolved.parents:
            raise ValueError("change paths must remain inside the project root.")
        if not any(
            resolved == directory or directory in resolved.parents
            for directory in self._allowed_dirs
        ):
            raise ValueError("change path is outside the allowed scope.")
        return relative


def _validated_relative(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("paths must be non-empty strings.")
    candidate = Path(raw.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("paths must be relative and cannot escape the root.")
    if candidate.suffix == ".env" or ".env" in candidate.name:
        raise ValueError("secret files are outside the sanctioned scope.")
    return candidate.as_posix()


def _normalized_paths(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        values = tuple(values)
    normalized: list[str] = []
    for value in values:
        normalized.append(_validated_relative(value))
    if not normalized:
        raise ValueError(f"{field_name} requires at least one path.")
    return tuple(dict.fromkeys(normalized))


def _display_path(raw: str) -> str:
    try:
        return _validated_relative(raw)
    except ValueError:
        return "invalid_path"


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0][:120] if text.strip() else "error"


# ---------------------------------------------------------------------------
# Model roles: worker (GLM 5.3 Flash), reviewer (Gemini), local (Ollama).
# Everything resolves through the existing ModelManager / provider registry.
# No key, endpoint or model name is hardcoded here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutonomousModelRoles:
    """Resolved model descriptors for the three minimal autonomy roles."""

    worker: ModelDescriptor | None
    reviewer: ModelDescriptor | None
    local: ModelDescriptor | None


def resolve_model_roles(
    model_manager: ModelManager | None = None,
    *,
    worker_id: str | None = None,
    reviewer_id: str | None = None,
    local_id: str | None = None,
) -> AutonomousModelRoles:
    """Resolve role models from explicit ids, environment, or task defaults."""
    manager = model_manager or ModelManager()
    worker = _resolve_role_model(
        manager,
        worker_id if worker_id is not None else os.getenv(WORKER_ROLE_ENV, ""),
        task="coding",
    )
    reviewer = _resolve_role_model(
        manager,
        reviewer_id
        if reviewer_id is not None
        else os.getenv(REVIEWER_ROLE_ENV, "") or DEFAULT_REVIEWER_LOGICAL_ID,
        task=None,
    )
    local = _resolve_role_model(
        manager,
        local_id
        if local_id is not None
        else os.getenv(LOCAL_ROLE_ENV, "") or DEFAULT_LOCAL_LOGICAL_ID,
        task=None,
    )
    return AutonomousModelRoles(worker=worker, reviewer=reviewer, local=local)


def _resolve_role_model(
    manager: ModelManager, identifier: str, *, task: str | None
) -> ModelDescriptor | None:
    if identifier.strip():
        return manager.resolve_model(identifier)
    if task is None:
        return None
    selection = manager.select_model(ModelSelectionRequest(task=task))
    return selection.descriptor if selection.success else None


WORKER_SYSTEM_PROMPT = """
Eres el planificador del runner autónomo de Atlas.

Devuelve EXCLUSIVAMENTE un objeto JSON válido con esta forma exacta:
{"reasoning": "<análisis breve>", "changes": [{"path": "ruta/relativa.py", "content": "archivo completo"}]}

Reglas:
- Solo puedes modificar archivos dentro del alcance permitido indicado.
- Nunca toques archivos .env, secretos, configuración de seguridad ni supervisor.
- Cada change debe contener el archivo completo propuesto.
- Si no necesitas cambios en esta iteración, devuelve "changes": [].
- No inventes APIs ni dependencias nuevas.
"""

REVIEWER_SYSTEM_PROMPT = """
Eres el revisor de segundo nivel del runner autónomo de Atlas.
Tu única función es dar una segunda opinión cuando la verificación
determinista resulta inconclusa.
Responde únicamente APPROVE o REJECT en la primera línea.
"""


class ModelPlanner:
    """Worker role: bounded plan generation through the configured provider."""

    def __init__(
        self,
        provider: object,
        model: str,
        *,
        provider_id: str | None = None,
        max_changes: int = MAX_PLAN_CHANGES,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("ModelPlanner requires a model name.")
        if provider is None:
            raise ValueError("ModelPlanner requires a chat inference provider.")
        self._provider = provider
        self._model = model.strip()
        self._provider_id = provider_id
        self._max_changes = max_changes

    def __call__(
        self, goal: str, iteration: int, history: Sequence[AutonomousIterationRecord]
    ) -> AutonomousPlan:
        recent = _history_prompt_lines(history)
        prompt = "\n".join(
            (
                f"Objetivo: {goal}",
                "Alcance permitido (solo rutas relativas dentro de él): indicado en la tarea.",
                f"Iteración actual: {iteration}",
                "Historial reciente (iteración | resultado | motivo):",
                *recent,
            )
        )
        response = self._provider.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": WORKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        payload = _parse_json_payload(_message_content(response))
        reasoning = payload.get("reasoning")
        raw_changes = payload.get("changes")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("planner payload requires a non-empty reasoning string.")
        if not isinstance(raw_changes, list):
            raise ValueError("planner payload requires a changes list.")
        if len(raw_changes) > self._max_changes:
            raise ValueError("planner payload exceeds the safe change limit.")
        changes = tuple(
            AutonomousFileChange(
                relative_path=_payload_text(entry, "path"),
                content=_payload_text(entry, "content"),
            )
            for entry in raw_changes
            if isinstance(entry, dict)
        )
        if len(changes) != len(raw_changes):
            raise ValueError("planner payload contains malformed change entries.")
        return AutonomousPlan(reasoning=reasoning.strip(), changes=changes)


class ModelReviewer:
    """Reviewer role: second opinion through the configured provider."""

    def __init__(self, provider: object, model: str, *, provider_id: str | None = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("ModelReviewer requires a model name.")
        if provider is None:
            raise ValueError("ModelReviewer requires a chat inference provider.")
        self._provider = provider
        self._model = model.strip()
        self._provider_id = provider_id

    def __call__(self, evaluation: GoalEvaluation) -> bool:
        test = evaluation.test_result
        prompt = "\n".join(
            (
                f"Evaluación del verificador: {evaluation.status.value} / {evaluation.reason}",
                f"Evidencia: {evaluation.verification.message or '(sin mensaje)'}",
                "Tests: "
                + (
                    "no ejecutados"
                    if test is None
                    else ("PASS" if test.passed else "FAIL")
                ),
                (
                    ""
                    if test is None or not test.output_tail
                    else "Salida de tests (recortada):\n" + test.output_tail[-800:]
                ),
            )
        )
        response = self._provider.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return _message_content(response).strip().upper().startswith("APPROVE")


def _payload_text(entry: dict, key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"planner change entry requires a non-empty {key}.")
    return value


def _history_prompt_lines(history: Sequence[AutonomousIterationRecord]) -> list[str]:
    lines: list[str] = []
    for record in history[-MAX_HISTORY_ITEMS_IN_PROMPT:]:
        lines.append(
            f"- iteración {record.iteration} | {record.outcome} | "
            f"{record.evaluation_status} | {record.evaluation_reason}"
        )
        if record.changed_paths:
            lines.append(f"  changed_paths: {', '.join(record.changed_paths)}")
        if record.test_passed is False:
            lines.append(f"  exit_code: {record.test_exit_code}")
            lines.append(f"  timed_out: {record.test_timed_out}")
            tail = record.test_output_tail.strip()
            if tail:
                lines.append("  Salida de tests (recortada):")
                lines.extend(
                    f"    {line}" for line in tail[-MAX_TEST_OUTPUT_CHARS_IN_PROMPT:].splitlines()
                )
    return lines or ["- (sin historial)"]


def _message_content(response: object) -> str:
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = response.get("content")
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else getattr(response, "content", None)
    if not isinstance(content, str):
        raise ValueError("provider returned a non-text response.")
    return content


def _parse_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("planner response does not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("planner response must be a JSON object.")
    return payload


def provider_for_descriptor(descriptor: ModelDescriptor) -> object:
    """Build one provider from the existing registry, env-configured only."""
    from models.chat_inference import default_provider_registry

    timeout = _env_float("ATLAS_OLLAMA_TIMEOUT", 120.0)
    keep_alive = os.getenv("ATLAS_OLLAMA_KEEP_ALIVE", "10m").strip() or "10m"
    return default_provider_registry(
        timeout=timeout,
        keep_alive=keep_alive,
        provider_id=descriptor.provider_id,
    ).get(descriptor.provider_id)


def worker_planner_from_roles(
    roles: AutonomousModelRoles, *, model_manager: ModelManager | None = None
) -> ModelPlanner:
    """Build the default GLM worker planner from the resolved roles."""
    descriptor = roles.worker
    if descriptor is None:
        raise AutonomousRunnerError(
            "no worker model available; configure ATLAS_AUTONOMY_WORKER_MODEL "
            "or register the descriptor in the model registry."
        )
    return ModelPlanner(
        provider_for_descriptor(descriptor),
        descriptor.model_name,
        provider_id=descriptor.provider_id,
    )


def reviewer_from_roles(
    roles: AutonomousModelRoles, *, model_manager: ModelManager | None = None
) -> ModelReviewer | None:
    """Build the Gemini reviewer; None when the role is not configured."""
    descriptor = roles.reviewer
    if descriptor is None:
        return None
    try:
        provider = provider_for_descriptor(descriptor)
    except (ValueError, RuntimeError):
        return None
    return ModelReviewer(
        provider, descriptor.model_name, provider_id=descriptor.provider_id
    )


def local_planner_from_roles(
    roles: AutonomousModelRoles, *, model_manager: ModelManager | None = None
) -> ModelPlanner | None:
    """Optional cheap local (Ollama) planner for simple tasks."""
    descriptor = roles.local
    if descriptor is None:
        return None
    try:
        provider = provider_for_descriptor(descriptor)
    except (ValueError, RuntimeError):
        return None
    return ModelPlanner(
        provider, descriptor.model_name, provider_id=descriptor.provider_id
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# ---------------------------------------------------------------------------
# Minimal development entrypoint: python -m core.autonomous_task_runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntrypointComponents:
    """Prebuilt collaborators injected by tests; None test_runner means default."""

    planner: AutonomousPlanner
    reviewer: AutonomousReviewer | None
    test_runner: PytestRunner | None


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m core.autonomous_task_runner",
        description=(
            "Ejecuta una tarea autónoma acotada (GOAL → PLAN → MODIFY → TEST → "
            "EVALUATE) con el AutonomousTaskRunner existente."
        ),
    )
    parser.add_argument("goal", help="Objetivo de la tarea en lenguaje natural.")
    parser.add_argument(
        "--allowed-paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Rutas relativas únicas permitidas para modificar (obligatorio).",
    )
    parser.add_argument(
        "--test-paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Rutas de pytest que validan el objetivo (obligatorio).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help=f"Iteraciones máximas, entre 1 y {MAX_AUTONOMOUS_ITERATIONS} (por defecto 5).",
    )
    parser.add_argument(
        "--worker",
        default=None,
        help=(
            "Identificador lógico del modelo worker; por defecto usa "
            f"{WORKER_ROLE_ENV} desde la configuración existente."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Raíz del proyecto sobre la que opera la tarea (por defecto el cwd).",
    )
    return parser


def _load_project_dotenv(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_root / ".env")


def _model_manager_from_environment() -> ModelManager:
    from core.model_registry import load_model_descriptors_from_environment

    return ModelManager(
        descriptors=load_model_descriptors_from_environment(
            reserved_logical_ids=(
                item.logical_id for item in ModelManager._DEFAULT_DESCRIPTORS
            ),
        ),
    )


def _default_components(
    worker_id: str | None, *, model_manager: ModelManager | None = None
) -> EntrypointComponents:
    manager = model_manager if model_manager is not None else _model_manager_from_environment()
    roles = resolve_model_roles(manager, worker_id=worker_id)
    planner = worker_planner_from_roles(roles, model_manager=manager)
    reviewer = reviewer_from_roles(roles, model_manager=manager)
    return EntrypointComponents(planner=planner, reviewer=reviewer, test_runner=None)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _error_payload(reason: str) -> dict[str, object]:
    return {"status": AutonomousRunnerStatus.BLOCKED.value, "reason": reason, "iterations": []}


def _execute(
    args: argparse.Namespace,
    *,
    project_root: Path,
    components: EntrypointComponents | None = None,
    model_manager: ModelManager | None = None,
) -> int:
    if not project_root.is_dir():
        _emit(_error_payload(f"project_root_does_not_exist:{project_root}"))
        return 2
    try:
        config = AutonomousTaskConfig(
            goal=args.goal,
            allowed_paths=args.allowed_paths,
            test_paths=args.test_paths,
            max_iterations=args.max_iterations,
        )
    except ValueError as error:
        _emit(_error_payload(f"invalid_config:{_first_line(str(error))}"))
        return 2
    if components is None:
        try:
            components = _default_components(args.worker, model_manager=model_manager)
        except AutonomousRunnerError as error:
            _emit(_error_payload(f"worker_unavailable:{_first_line(str(error))}"))
            return 2
    runner = AutonomousTaskRunner(
        project_root,
        config,
        planner=components.planner,
        reviewer=components.reviewer,
        test_runner=components.test_runner,
    )
    result = runner.run()
    _emit(_result_payload(result))
    return 0 if result.success else 1


def _result_payload(result: AutonomousTaskResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "reason": result.reason,
        "rolled_back": result.rolled_back,
        "last_test_passed": (
            None if result.last_test_result is None else result.last_test_result.passed
        ),
        "checkpoint_events": list(result.checkpoint_events),
        "iterations": [_iteration_payload(record) for record in result.iterations],
    }


def _iteration_payload(record: AutonomousIterationRecord) -> dict[str, object]:
    return {
        "iteration": record.iteration,
        "outcome": record.outcome,
        "reasoning": record.reasoning,
        "changed_paths": list(record.changed_paths),
        "evaluation_status": record.evaluation_status,
        "evaluation_reason": record.evaluation_reason,
        "test_passed": record.test_passed,
        "test_exit_code": record.test_exit_code,
        "test_timed_out": record.test_timed_out,
        "test_output_tail": record.test_output_tail,
        "reviewer_consulted": record.reviewer_consulted,
        "reviewer_approved": record.reviewer_approved,
        "restored": record.restored,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    _load_project_dotenv(project_root)
    return _execute(args, project_root=project_root)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
