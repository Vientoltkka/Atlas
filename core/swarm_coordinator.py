"""Controlled cognitive swarm coordination subordinated to the Orchestrator.

V1 scope: read-only, cognitive tasks only. Workers are bounded executions of
existing registered agents through the existing model/provider selection
infrastructure. The coordinator never executes tools, never authorizes
critical actions, and never answers the user directly: it returns exactly one
structured SwarmResult to the Orchestrator.

Worker threads are daemon threads; when a worker or the global timeout
expires, the pending result is discarded and the abandoned thread terminates
on its own, bounded by the provider HTTP timeout (same abandonment pattern as
core/skill_executor.py). A closed allowlist keeps workers on agents whose
``run()`` is demonstrably cognitive/read-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from threading import Semaphore, Thread
import time
from types import MappingProxyType

from agents.base_agent import AgentResponse
from agents.registry import AgentRegistry
from core.model_health import ModelHealthChecker
from core.model_inference import ModelInferenceRunner
from core.model_selection_policy import ModelSelectionPolicy


MAX_SWARM_WORKERS = 8
MAX_SWARM_STRING_LENGTH = 4_000
MAX_SWARM_METADATA_ITEMS = 16
MAX_SWARM_CONSTRAINTS = 16
MAX_WORKER_ERROR_LENGTH = 500
DEFAULT_WORKER_TIMEOUT_SECONDS = 60
DEFAULT_SWARM_TIMEOUT_SECONDS = 120

DEFAULT_WORKER_SYSTEM_PROMPT = (
    "Eres un worker de un swarm controlado por Atlas. Trabaja de forma "
    "cognitiva y de solo lectura: investiga, analiza, compara o planifica. "
    "No ejecutes acciones destructivas ni modifiques archivos o sistemas."
)

DEFAULT_SWARM_AGENT_ALLOWLIST = frozenset(
    {
        "chat",
        "code",
        "training",
        "nutrition",
        "medical",
        "legal",
        "finance",
        "project",
    }
)


class SwarmError(RuntimeError):
    """Base error for controlled swarm coordination."""


class InvalidSwarmTaskError(SwarmError):
    """Raised when a swarm task or worker spec is malformed."""


class SwarmStatus(str, Enum):
    """Terminal statuses for one swarm task."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"


class SwarmWorkerStatus(str, Enum):
    """Terminal statuses for one swarm worker execution."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    AGENT_BLOCKED = "AGENT_BLOCKED"


class SwarmCriticStatus(str, Enum):
    """Terminal statuses for the swarm critic step."""

    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class SwarmPolicy:
    """Bounded policy for swarm execution. Concurrency is capped and opt-in.

    ``allowed_agent_names`` is a closed allowlist of agents whose ``run()`` is
    cognitive/read-only. Agents outside the allowlist (e.g. ``coding``, which
    holds write-capable collaborators and pending-change state) are refused
    before execution. Passing ``None`` keeps the safe default.
    """

    max_workers: int = 4
    worker_timeout_seconds: int = DEFAULT_WORKER_TIMEOUT_SECONDS
    global_timeout_seconds: int = DEFAULT_SWARM_TIMEOUT_SECONDS
    allow_partial: bool = True
    allowed_agent_names: frozenset[str] = DEFAULT_SWARM_AGENT_ALLOWLIST

    def __post_init__(self) -> None:
        for name in ("max_workers", "worker_timeout_seconds", "global_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidSwarmTaskError(f"{name} must be a positive integer.")
        if self.max_workers > MAX_SWARM_WORKERS:
            raise InvalidSwarmTaskError(f"max_workers cannot exceed {MAX_SWARM_WORKERS}.")
        if not isinstance(self.allow_partial, bool):
            raise InvalidSwarmTaskError("allow_partial must be a bool.")
        if not isinstance(self.allowed_agent_names, frozenset):
            raise InvalidSwarmTaskError("allowed_agent_names must be a frozenset of agent names.")
        for agent_name in self.allowed_agent_names:
            _identifier(agent_name, "allowed_agent_names entry")


@dataclass(frozen=True, slots=True)
class SwarmWorkerSpec:
    """One bounded worker: an existing agent executed in isolation."""

    worker_id: str
    agent_name: str
    objective: str | None = None
    context: Mapping[str, object] | None = None
    model_task: str = "chat"
    preferred_model_id: str | None = None
    timeout_seconds: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _identifier(self.worker_id, "worker_id"))
        object.__setattr__(self, "agent_name", _identifier(self.agent_name, "agent_name"))
        if self.objective is not None:
            object.__setattr__(self, "objective", _text(self.objective, "objective"))
        if self.context is not None:
            object.__setattr__(self, "context", MappingProxyType(_safe_mapping(self.context, "context")))
        object.__setattr__(self, "model_task", _identifier(self.model_task, "model_task"))
        if self.preferred_model_id is not None:
            object.__setattr__(self, "preferred_model_id", _identifier(self.preferred_model_id, "preferred_model_id"))
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
                raise InvalidSwarmTaskError("timeout_seconds must be a positive integer when provided.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))


@dataclass(frozen=True, slots=True)
class SwarmTask:
    """One cognitive swarm task received from the Orchestrator."""

    task_id: str
    objective: str
    worker_specs: tuple[SwarmWorkerSpec, ...]
    context: Mapping[str, object] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    timeout_seconds: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(self, "objective", _text(self.objective, "objective"))
        if not isinstance(self.worker_specs, tuple) or not self.worker_specs:
            raise InvalidSwarmTaskError("worker_specs must be a non-empty tuple of SwarmWorkerSpec.")
        for spec in self.worker_specs:
            if not isinstance(spec, SwarmWorkerSpec):
                raise InvalidSwarmTaskError("worker_specs must contain only SwarmWorkerSpec instances.")
        worker_ids = tuple(spec.worker_id for spec in self.worker_specs)
        if len(set(worker_ids)) != len(worker_ids):
            raise InvalidSwarmTaskError("worker_ids must be unique within a task.")
        object.__setattr__(self, "context", MappingProxyType(_safe_mapping(self.context, "context")))
        object.__setattr__(self, "constraints", _text_tuple(self.constraints, "constraints", MAX_SWARM_CONSTRAINTS))
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
                raise InvalidSwarmTaskError("timeout_seconds must be a positive integer when provided.")
        object.__setattr__(self, "metadata", MappingProxyType(_safe_mapping(self.metadata, "metadata")))


@dataclass(frozen=True, slots=True)
class SwarmWorkerResult:
    """Independent result of one swarm worker."""

    worker_id: str
    agent_id: str
    status: SwarmWorkerStatus
    result: str | None = None
    logical_model_id: str | None = None
    provider_id: str | None = None
    physical_model_name: str | None = None
    confidence: float | None = None
    error: str | None = None
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _identifier(self.worker_id, "worker_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if not isinstance(self.status, SwarmWorkerStatus):
            object.__setattr__(self, "status", SwarmWorkerStatus(self.status))
        if self.result is not None:
            object.__setattr__(self, "result", _text(self.result, "result"))
        for name in ("logical_model_id", "provider_id", "physical_model_name"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(str(value), name))
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise InvalidSwarmTaskError("confidence must be a number or None.")
            object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        if self.error is not None:
            object.__setattr__(self, "error", _text(self.error[:MAX_WORKER_ERROR_LENGTH], "error"))
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, (int, float)) or self.duration_seconds < 0:
            raise InvalidSwarmTaskError("duration_seconds must be a non-negative number.")
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))

    @property
    def succeeded(self) -> bool:
        return self.status is SwarmWorkerStatus.SUCCEEDED


@dataclass(frozen=True, slots=True)
class SwarmCriticResult:
    """Structured output of the critic/evaluator step."""

    status: SwarmCriticStatus
    findings: tuple[str, ...] = ()
    disagreement_worker_ids: tuple[str, ...] = ()
    confidence: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SwarmCriticStatus):
            object.__setattr__(self, "status", SwarmCriticStatus(self.status))
        object.__setattr__(self, "findings", _text_tuple(self.findings, "findings", MAX_SWARM_CONSTRAINTS))
        object.__setattr__(
            self,
            "disagreement_worker_ids",
            _text_tuple(self.disagreement_worker_ids, "disagreement_worker_ids", MAX_SWARM_WORKERS),
        )
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise InvalidSwarmTaskError("confidence must be a number or None.")
            object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        if self.error is not None:
            object.__setattr__(self, "error", _text(self.error[:MAX_WORKER_ERROR_LENGTH], "error"))


@dataclass(frozen=True, slots=True)
class SwarmEvent:
    """Safe observability event for swarm execution."""

    name: str
    status: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "event name"))
        if not isinstance(self.status, str) or not self.status.strip():
            raise InvalidSwarmTaskError("event status must be a non-empty string.")
        object.__setattr__(self, "status", self.status.strip())
        object.__setattr__(self, "details", MappingProxyType(_safe_mapping(self.details, "details")))


@dataclass(frozen=True, slots=True)
class SwarmResult:
    """The single structured result returned to the Orchestrator."""

    task_id: str
    status: SwarmStatus
    worker_results: tuple[SwarmWorkerResult, ...] = ()
    critic_result: SwarmCriticResult | None = None
    final_synthesis: str | None = None
    confidence: float = 0.0
    events: tuple[SwarmEvent, ...] = ()
    metrics: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    safe_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        if not isinstance(self.status, SwarmStatus):
            object.__setattr__(self, "status", SwarmStatus(self.status))
        object.__setattr__(self, "worker_results", tuple(self.worker_results))
        if self.final_synthesis is not None:
            object.__setattr__(self, "final_synthesis", _text(self.final_synthesis, "final_synthesis"))
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise InvalidSwarmTaskError("confidence must be a number between 0 and 1.")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


CriticCallable = Callable[[SwarmTask, tuple[SwarmWorkerResult, ...]], SwarmCriticResult]
SynthesizerCallable = Callable[[SwarmTask, tuple[SwarmWorkerResult, ...], SwarmCriticResult | None], str]


class SwarmCoordinator:
    """Coordinate bounded cognitive workers, critic, and synthesis.

    The coordinator is subordinated to the Orchestrator: it receives one task,
    runs read-only workers through the existing model/provider selection
    infrastructure, tolerates partial failure, and returns exactly one
    SwarmResult. It holds no tool registry and cannot execute tools.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        model_manager,
        model_selection_policy: ModelSelectionPolicy | None = None,
        health_checker: ModelHealthChecker | None = None,
        *,
        critic: CriticCallable | None = None,
        synthesizer: SynthesizerCallable | None = None,
        policy: SwarmPolicy | None = None,
    ) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise SwarmError("SwarmCoordinator requires agents.registry.AgentRegistry.")
        if not isinstance(model_selection_policy, ModelSelectionPolicy) and model_selection_policy is not None:
            raise SwarmError("model_selection_policy must be ModelSelectionPolicy or None.")
        self._agent_registry = agent_registry
        self._model_manager = model_manager
        self._model_selection_policy = model_selection_policy or ModelSelectionPolicy()
        self._health_checker = health_checker
        self._critic = critic
        self._synthesizer = synthesizer
        self._policy = policy or SwarmPolicy()
        self._concurrency = Semaphore(self._policy.max_workers)

    @property
    def policy(self) -> SwarmPolicy:
        return self._policy

    def execute(self, task: SwarmTask) -> SwarmResult:
        """Execute one swarm task and return exactly one structured result."""

        if not isinstance(task, SwarmTask):
            return SwarmResult(
                task_id="invalid",
                status=SwarmStatus.INVALID_REQUEST,
                error_code="INVALID_REQUEST",
                safe_message="task must be SwarmTask.",
            )
        events: list[SwarmEvent] = []
        _event(events, "swarm_execution_requested", "started", {"task_id": task.task_id, "workers": len(task.worker_specs)})

        if len(task.worker_specs) > self._policy.max_workers:
            _event(events, "swarm_execution_rejected", "failed", {"reason": "max_workers_exceeded"})
            return SwarmResult(
                task_id=task.task_id,
                status=SwarmStatus.INVALID_REQUEST,
                events=tuple(events),
                metrics=_metrics(requested=1, invalid=1),
                error_code="MAX_WORKERS_EXCEEDED",
                safe_message="task exceeds the configured max_workers limit.",
            )

        global_timeout = task.timeout_seconds or self._policy.global_timeout_seconds
        deadline = time.monotonic() + min(global_timeout, self._policy.global_timeout_seconds)

        worker_results = self._execute_workers(task, deadline, events)
        succeeded = tuple(result for result in worker_results if result.succeeded)
        failed = tuple(result for result in worker_results if not result.succeeded)

        if not succeeded:
            _event(events, "swarm_execution_failed", "failed", {"reason": "no_worker_succeeded"})
            return SwarmResult(
                task_id=task.task_id,
                status=SwarmStatus.FAILED,
                worker_results=tuple(worker_results),
                confidence=0.0,
                events=tuple(events),
                metrics=_metrics(
                    requested=1,
                    failed=1,
                    workers_started=len(task.worker_specs),
                    workers_succeeded=0,
                    workers_failed=len(failed),
                    workers_timeout=sum(1 for result in failed if result.status is SwarmWorkerStatus.TIMEOUT),
                ),
                error_code="NO_WORKER_SUCCEEDED",
                safe_message="ningún worker del swarm produjo un resultado válido.",
            )

        critic_result = self._run_critic(task, tuple(worker_results), events)
        synthesis = self._run_synthesizer(task, tuple(worker_results), critic_result, events)
        confidence = len(succeeded) / len(worker_results)
        status = SwarmStatus.SUCCESS if not failed else SwarmStatus.PARTIAL_SUCCESS
        _event(events, "swarm_execution_completed", "finished", {"status": status.value})
        return SwarmResult(
            task_id=task.task_id,
            status=status,
            worker_results=tuple(worker_results),
            critic_result=critic_result,
            final_synthesis=synthesis,
            confidence=confidence,
            events=tuple(events),
            metrics=_metrics(
                requested=1,
                succeeded=1 if status is SwarmStatus.SUCCESS else 0,
                partial=1 if status is SwarmStatus.PARTIAL_SUCCESS else 0,
                workers_started=len(task.worker_specs),
                workers_succeeded=len(succeeded),
                workers_failed=len(failed),
                workers_timeout=sum(1 for result in failed if result.status is SwarmWorkerStatus.TIMEOUT),
            ),
        )

    # ------------------------------------------------------------------
    # Worker execution
    # ------------------------------------------------------------------

    def _execute_workers(
        self,
        task: SwarmTask,
        deadline: float,
        events: list[SwarmEvent],
    ) -> list[SwarmWorkerResult]:
        outcomes: dict[str, Queue[SwarmWorkerResult]] = {}
        threads: list[Thread] = []
        for spec in task.worker_specs:
            queue: Queue[SwarmWorkerResult] = Queue(maxsize=1)
            outcomes[spec.worker_id] = queue
            thread = Thread(
                target=self._worker_target,
                args=(task, spec, queue),
                name=f"atlas-swarm-{task.task_id}-{spec.worker_id}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            _event(events, "swarm_worker_started", "started", {"worker_id": spec.worker_id, "agent_id": spec.agent_name})

        results: list[SwarmWorkerResult] = []
        for spec in task.worker_specs:
            worker_timeout = (
                spec.timeout_seconds
                or task.timeout_seconds
                or self._policy.worker_timeout_seconds
            )
            effective_timeout = min(worker_timeout, deadline - time.monotonic())
            if effective_timeout <= 0:
                results.append(self._collect_expired(spec, outcomes[spec.worker_id]))
                continue
            try:
                results.append(outcomes[spec.worker_id].get(timeout=effective_timeout))
            except Empty:
                # El hilo pudo terminar durante la recolección anterior: drena
                # cualquier resultado ya disponible antes de declarar TIMEOUT.
                results.append(self._collect_expired(spec, outcomes[spec.worker_id]))
        return results

    @staticmethod
    def _collect_expired(
        spec: SwarmWorkerSpec,
        queue: Queue[SwarmWorkerResult],
    ) -> SwarmWorkerResult:
        """Drain one already-finished worker result, else mark it TIMEOUT."""
        try:
            return queue.get_nowait()
        except Empty:
            return SwarmWorkerResult(
                spec.worker_id,
                spec.agent_name,
                SwarmWorkerStatus.TIMEOUT,
                error="worker exceeded its effective timeout.",
            )

    def _worker_target(
        self,
        task: SwarmTask,
        spec: SwarmWorkerSpec,
        queue: Queue[SwarmWorkerResult],
    ) -> None:
        with self._concurrency:
            queue.put(self._run_worker(task, spec))

    def _run_worker(self, task: SwarmTask, spec: SwarmWorkerSpec) -> SwarmWorkerResult:
        started = time.monotonic()
        if spec.agent_name not in self._policy.allowed_agent_names:
            return SwarmWorkerResult(
                spec.worker_id,
                spec.agent_name,
                SwarmWorkerStatus.AGENT_BLOCKED,
                error=f"agent '{spec.agent_name}' is outside the swarm cognitive allowlist.",
                duration_seconds=time.monotonic() - started,
            )
        agent = self._agent_registry.get(spec.agent_name)
        if agent is None or not callable(getattr(agent, "run", None)):
            return SwarmWorkerResult(
                spec.worker_id,
                spec.agent_name,
                SwarmWorkerStatus.AGENT_UNAVAILABLE,
                error=f"agent '{spec.agent_name}' is not registered.",
                duration_seconds=time.monotonic() - started,
            )

        objective = spec.objective or task.objective
        messages: list[dict[str, str]] = [
            {"role": "system", "content": DEFAULT_WORKER_SYSTEM_PROMPT},
            {"role": "user", "content": _worker_user_message(task, spec, objective)},
        ]
        runner = ModelInferenceRunner(self._model_manager, health_checker=self._health_checker)
        captured: dict[str, str | None] = {"model": None, "provider": None}

        def infer(selected_model: str, selected_provider_id: str | None) -> str:
            captured["model"] = selected_model
            captured["provider"] = selected_provider_id
            raw = agent.run(
                model=selected_model,
                messages=messages,
                provider_id=selected_provider_id,
            )
            return raw.text if isinstance(raw, AgentResponse) else raw

        try:
            text = runner.run(
                self._model_selection_policy.create_request(
                    task=spec.model_task,
                    preferred_model_id=spec.preferred_model_id,
                ),
                infer,
            )
        except Exception as error:  # noqa: BLE001 - one worker failure must not destroy the swarm
            return SwarmWorkerResult(
                spec.worker_id,
                spec.agent_name,
                SwarmWorkerStatus.FAILED,
                error=str(error) or type(error).__name__,
                duration_seconds=time.monotonic() - started,
            )

        inference = runner.last_result
        return SwarmWorkerResult(
            spec.worker_id,
            spec.agent_name,
            SwarmWorkerStatus.SUCCEEDED,
            result=text,
            logical_model_id=(
                inference.final_logical_model_id
                if inference is not None
                else captured.get("model")
            ),
            provider_id=captured.get("provider"),
            physical_model_name=(
                inference.final_physical_model_name
                if inference is not None
                else captured.get("model")
            ),
            duration_seconds=time.monotonic() - started,
        )

    # ------------------------------------------------------------------
    # Critic and synthesis
    # ------------------------------------------------------------------

    def _run_critic(
        self,
        task: SwarmTask,
        worker_results: tuple[SwarmWorkerResult, ...],
        events: list[SwarmEvent],
    ) -> SwarmCriticResult:
        _event(events, "swarm_critic_started", "started", {"results": len(worker_results)})
        critic = self._critic or default_swarm_critic
        try:
            critic_result = critic(task, worker_results)
            if not isinstance(critic_result, SwarmCriticResult):
                raise SwarmError("critic must return SwarmCriticResult.")
        except Exception as error:  # noqa: BLE001 - critic failure must not destroy valid results
            _event(events, "swarm_critic_failed", "failed")
            return SwarmCriticResult(
                SwarmCriticStatus.FAILED,
                findings=("critic execution failed; synthesis preserves raw uncertainty.",),
                error=str(error) or type(error).__name__,
            )
        _event(events, "swarm_critic_finished", "finished", {"status": critic_result.status.value})
        return critic_result

    def _run_synthesizer(
        self,
        task: SwarmTask,
        worker_results: tuple[SwarmWorkerResult, ...],
        critic_result: SwarmCriticResult | None,
        events: list[SwarmEvent],
    ) -> str:
        _event(events, "swarm_synthesis_started", "started")
        synthesizer = self._synthesizer or default_swarm_synthesizer
        try:
            synthesis = synthesizer(task, worker_results, critic_result)
        except Exception as error:  # noqa: BLE001 - fall back to deterministic synthesis
            synthesis = default_swarm_synthesizer(task, worker_results, critic_result)
            synthesis = f"{synthesis}\n(Sintetizador principal falló: {type(error).__name__}.)"
        _event(events, "swarm_synthesis_finished", "finished")
        return synthesis


# ----------------------------------------------------------------------
# Default deterministic critic and synthesizer
# ----------------------------------------------------------------------


def default_swarm_critic(
    task: SwarmTask,
    worker_results: tuple[SwarmWorkerResult, ...],
) -> SwarmCriticResult:
    """Deterministic read-only review of worker outputs and failures."""

    findings: list[str] = []
    succeeded = tuple(result for result in worker_results if result.succeeded)
    failed = tuple(result for result in worker_results if not result.succeeded)
    for result in failed:
        findings.append(
            f"worker '{result.worker_id}' terminó con estado {result.status.value}: {result.error or 'sin detalle'}."
        )
    for result in succeeded:
        if not result.result or not result.result.strip():
            findings.append(f"worker '{result.worker_id}' devolvió una respuesta vacía o incompleta.")
    disagreement_ids: tuple[str, ...] = ()
    if len(succeeded) >= 2:
        texts = {result.result for result in succeeded}
        if len(texts) > 1:
            disagreement_ids = tuple(result.worker_id for result in succeeded)
            findings.append(
                "los workers produjeron respuestas divergentes; no se puede establecer consenso automáticamente."
            )
    if not findings:
        findings.append("sin contradicciones, omisiones o errores detectables de forma determinista.")
    confidence = len(succeeded) / len(worker_results) if worker_results else 0.0
    return SwarmCriticResult(
        SwarmCriticStatus.OK,
        findings=tuple(findings),
        disagreement_worker_ids=disagreement_ids,
        confidence=confidence,
    )


def default_swarm_synthesizer(
    task: SwarmTask,
    worker_results: tuple[SwarmWorkerResult, ...],
    critic_result: SwarmCriticResult | None,
) -> str:
    """Deterministic synthesis combining findings, disagreements, and failures."""

    succeeded = tuple(result for result in worker_results if result.succeeded)
    failed = tuple(result for result in worker_results if not result.succeeded)
    lines: list[str] = [f"Síntesis del swarm para la tarea '{task.task_id}':"]
    for index, result in enumerate(succeeded, start=1):
        lines.append(f"{index}. [{result.agent_id}] {result.result}")
    if critic_result is not None and critic_result.disagreement_worker_ids:
        lines.append(
            "Desacuerdos entre workers: "
            + ", ".join(critic_result.disagreement_worker_ids)
            + ". Se preserva la incertidumbre; no se inventa consenso."
        )
    for finding in critic_result.findings if critic_result is not None else ():
        lines.append(f"Revisión: {finding}")
    if failed:
        failed_summary = ", ".join(f"{result.worker_id} ({result.status.value})" for result in failed)
        lines.append(f"Fallos parciales: {failed_summary}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _worker_user_message(task: SwarmTask, spec: SwarmWorkerSpec, objective: str) -> str:
    parts = [f"Objetivo: {objective}"]
    if task.constraints:
        parts.append("Restricciones: " + "; ".join(task.constraints))
    context = spec.context or task.context
    if context:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(context.items(), key=lambda item: str(item[0])))
        parts.append(f"Contexto: {rendered}")
    return "\n".join(parts)


def _event(
    events: list[SwarmEvent],
    name: str,
    status: str,
    details: Mapping[str, object] | None = None,
) -> None:
    events.append(SwarmEvent(name, status, {} if details is None else details))


def _metrics(
    *,
    requested: int = 0,
    succeeded: int = 0,
    partial: int = 0,
    failed: int = 0,
    invalid: int = 0,
    workers_started: int = 0,
    workers_succeeded: int = 0,
    workers_failed: int = 0,
    workers_timeout: int = 0,
) -> dict[str, int]:
    return {
        "swarm_tasks_requested": requested,
        "swarm_tasks_succeeded": succeeded,
        "swarm_tasks_partial": partial,
        "swarm_tasks_failed": failed,
        "swarm_tasks_invalid": invalid,
        "swarm_workers_started": workers_started,
        "swarm_workers_succeeded": workers_succeeded,
        "swarm_workers_failed": workers_failed,
        "swarm_workers_timeout": workers_timeout,
    }


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSwarmTaskError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if len(normalized) > MAX_SWARM_STRING_LENGTH:
        raise InvalidSwarmTaskError(f"{field_name} is too long.")
    return normalized


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidSwarmTaskError(f"{field_name} must be a string.")
    if len(value) > MAX_SWARM_STRING_LENGTH:
        raise InvalidSwarmTaskError(f"{field_name} is too long.")
    return value


def _text_tuple(values: Iterable[str], field_name: str, limit: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise InvalidSwarmTaskError(f"{field_name} must be an iterable of strings.")
    normalized = tuple(_text(value, field_name) for value in values)
    if len(normalized) > limit:
        raise InvalidSwarmTaskError(f"{field_name} has too many items.")
    return normalized


def _safe_mapping(mapping: Mapping[str, object], field_name: str) -> dict[str, object]:
    if not isinstance(mapping, Mapping):
        raise InvalidSwarmTaskError(f"{field_name} must be a mapping.")
    if len(mapping) > MAX_SWARM_METADATA_ITEMS:
        raise InvalidSwarmTaskError(f"{field_name} has too many items.")
    safe: dict[str, object] = {}
    for key, value in mapping.items():
        normalized_key = _identifier(str(key), f"{field_name} key")
        if not isinstance(value, (bool, int, float, str)) and value is not None:
            raise InvalidSwarmTaskError(f"{field_name} values must be scalar or None.")
        safe[normalized_key] = value
    return safe
