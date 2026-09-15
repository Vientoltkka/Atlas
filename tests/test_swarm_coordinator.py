from __future__ import annotations

import time
from types import SimpleNamespace

from agents.base_agent import AgentResponse, BaseAgent
from agents.registry import AgentRegistry
from core.model_manager import ModelDescriptor, ModelManager
from core.model_selection_policy import ModelSelectionPolicy
from core.orchestrator import AtlasOrchestrator
from core.router import Router
from core.swarm_coordinator import (
    DEFAULT_SWARM_AGENT_ALLOWLIST,
    InvalidSwarmTaskError,
    SwarmCoordinator,
    SwarmCriticResult,
    SwarmCriticStatus,
    SwarmPolicy,
    SwarmResult,
    SwarmStatus,
    SwarmTask,
    SwarmWorkerResult,
    SwarmWorkerSpec,
    SwarmWorkerStatus,
    default_swarm_critic,
    default_swarm_synthesizer,
)
from memory.conversation import ConversationMemory


class StaticModelSource:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list_models(self) -> list[str]:
        return list(self._models)


class RecordingAgent(BaseAgent):
    """Minimal existing-agent-shaped double for swarm workers."""

    def __init__(self, name: str, answer: str, *, delay: float = 0.0, fail: bool = False) -> None:
        self._name = name
        self.answer = answer
        self.delay = delay
        self.fail = fail
        self.calls: list[tuple[str, str | None]] = []
        self.messages: list[list[dict[str, str]]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"recording agent {self._name}"

    def run(self, model: str, messages: list[dict[str, str]], *, provider_id: str | None = None) -> str | AgentResponse:
        self.calls.append((model, provider_id))
        self.messages.append(messages)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("worker boom")
        return self.answer


def _descriptor(logical_id: str, model_name: str, provider_id: str, *, fallbacks: tuple[str, ...] = ()) -> ModelDescriptor:
    return ModelDescriptor(
        logical_id=logical_id,
        provider_id=provider_id,
        model_name=model_name,
        capabilities=("chat",),
        fallback_logical_ids=fallbacks,
    )


def _manager() -> ModelManager:
    return ModelManager(
        StaticModelSource(["local:latest", "remote:latest"]),
        (
            _descriptor("swarm-local", "local:latest", "ollama"),
            _descriptor("swarm-remote", "remote:latest", "gemini", fallbacks=("swarm-local",)),
        ),
    )


def _coordinator(
    agents: tuple[RecordingAgent, ...],
    *,
    manager: ModelManager | None = None,
    critic=None,
    synthesizer=None,
    policy: SwarmPolicy | None = None,
) -> SwarmCoordinator:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    if policy is None:
        # Test convenience: allow exactly the default cognitive allowlist plus
        # the fake agents registered by this test module.
        policy = SwarmPolicy(
            allowed_agent_names=DEFAULT_SWARM_AGENT_ALLOWLIST
            | frozenset(agent.name for agent in agents)
        )
    return SwarmCoordinator(
        registry,
        manager if manager is not None else _manager(),
        model_selection_policy=ModelSelectionPolicy(),
        critic=critic,
        synthesizer=synthesizer,
        policy=policy,
    )


def _task(specs: tuple[SwarmWorkerSpec, ...], *, task_id: str = "swarm-test", timeout_seconds: int | None = None) -> SwarmTask:
    return SwarmTask(
        task_id=task_id,
        objective="analiza el estado del proyecto Atlas",
        worker_specs=specs,
        constraints=("solo lectura", "no modificar archivos"),
        timeout_seconds=timeout_seconds,
    )


def _spec(worker_id: str, agent_name: str, **kwargs) -> SwarmWorkerSpec:
    return SwarmWorkerSpec(worker_id=worker_id, agent_name=agent_name, **kwargs)


def _orchestrator(coordinator: SwarmCoordinator | None) -> AtlasOrchestrator:
    chat = RecordingAgent("chat", "respuesta")
    registry = SimpleNamespace(get=lambda name: chat if name == "chat" else None)
    return AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task=prompt, objective=prompt)),
        router=Router(),
        model_manager=SimpleNamespace(choose_model=lambda agent_name: f"model:{agent_name}"),
        memory=ConversationMemory(),
        registry=registry,
        write_file=SimpleNamespace(execute=lambda *_args: "written"),
        swarm_coordinator=coordinator,
    )


# 1. Orchestrator puede invocar SwarmCoordinator.
def test_orchestrator_can_invoke_swarm_coordinator() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "resultado a"),))
    orchestrator = _orchestrator(coordinator)
    result = orchestrator.execute_swarm(_task((_spec("w1", "chat"),), timeout_seconds=15))

    assert result.status in {SwarmStatus.SUCCESS, SwarmStatus.PARTIAL_SUCCESS}
    assert result.final_synthesis is not None


# 2. SwarmCoordinator devuelve UN resultado al Orchestrator.
def test_coordinator_returns_exactly_one_structured_result() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "ok"),))
    result = coordinator.execute(_task((_spec("w1", "chat"),), timeout_seconds=15))

    assert isinstance(result, SwarmResult)
    assert result.task_id == "swarm-test"
    assert result.status is SwarmStatus.SUCCESS
    assert result.final_synthesis is not None
    assert result.critic_result is not None


# 3. Dos o más workers producen resultados independientes.
def test_multiple_workers_produce_independent_results() -> None:
    agent_a = RecordingAgent("chat_a", "respuesta a")
    agent_b = RecordingAgent("chat_b", "respuesta b")
    coordinator = _coordinator((agent_a, agent_b))
    result = coordinator.execute(_task((_spec("w1", "chat_a"), _spec("w2", "chat_b")), timeout_seconds=15))

    assert result.status is SwarmStatus.SUCCESS
    assert len(result.worker_results) == 2
    assert result.worker_results[0].worker_id == "w1"
    assert result.worker_results[1].worker_id == "w2"
    assert agent_a.calls and agent_b.calls
    assert result.metrics["swarm_workers_succeeded"] == 2


# 4. Un worker puede fallar sin perder resultados válidos de los demás.
def test_one_worker_failure_preserves_valid_results() -> None:
    failing = RecordingAgent("bad_chat", "", fail=True)
    healthy = RecordingAgent("ok_chat", "respuesta válida")
    coordinator = _coordinator((failing, healthy))
    result = coordinator.execute(
        _task((_spec("w_bad", "bad_chat"), _spec("w_ok", "ok_chat")), timeout_seconds=15)
    )

    assert result.status is SwarmStatus.PARTIAL_SUCCESS
    statuses = {worker.worker_id: worker.status for worker in result.worker_results}
    assert statuses["w_bad"] is SwarmWorkerStatus.FAILED
    assert statuses["w_ok"] is SwarmWorkerStatus.SUCCEEDED
    assert result.worker_results[1].result == "respuesta válida"
    assert result.final_synthesis is not None


# 5. Timeout de worker no bloquea indefinidamente el swarm.
def test_worker_timeout_does_not_block_the_swarm() -> None:
    slow = RecordingAgent("slow_chat", "muy lento", delay=5.0)
    fast = RecordingAgent("fast_chat", "rápido")
    coordinator = _coordinator(
        (slow, fast),
        policy=SwarmPolicy(
            max_workers=2,
            worker_timeout_seconds=1,
            global_timeout_seconds=15,
            allowed_agent_names=DEFAULT_SWARM_AGENT_ALLOWLIST | frozenset({"slow_chat", "fast_chat"}),
        ),
    )
    started = time.monotonic()
    # El timeout de la tarea (3s) es el presupuesto efectivo del worker: por
    # debajo del delay del agente lento (5s), fuerza su TIMEOUT sin colgar el swarm.
    result = coordinator.execute(_task((_spec("w_fast", "fast_chat"), _spec("w_slow", "slow_chat")), timeout_seconds=3))
    elapsed = time.monotonic() - started

    statuses = {worker.worker_id: worker.status for worker in result.worker_results}
    assert statuses["w_slow"] is SwarmWorkerStatus.TIMEOUT
    assert statuses["w_fast"] is SwarmWorkerStatus.SUCCEEDED
    assert result.status is SwarmStatus.PARTIAL_SUCCESS
    assert elapsed < 5.0


# 6. model_id/provider_id se conserva correctamente.
def test_model_and_provider_pair_is_preserved() -> None:
    manager = ModelManager(
        StaticModelSource(["remote:latest"]),
        (_descriptor("swarm-gemini", "remote:latest", "gemini"),),
    )
    agent = RecordingAgent("chat", "respuesta")
    coordinator = _coordinator((agent,), manager=manager)
    result = coordinator.execute(
        _task(
            (
                SwarmWorkerSpec(
                    worker_id="w1",
                    agent_name="chat",
                    preferred_model_id="swarm-gemini",
                    timeout_seconds=15,
                ),
            ),
        )
    )

    worker = result.worker_results[0]
    assert result.status is SwarmStatus.SUCCESS
    assert worker.logical_model_id == "swarm-gemini"
    assert worker.provider_id == "gemini"
    assert worker.physical_model_name == "remote:latest"
    assert agent.calls[0] == ("remote:latest", "gemini")


# 7. Critic recibe los resultados correctos.
def test_critic_receives_worker_results() -> None:
    received: dict[str, object] = {}

    def critic(task: SwarmTask, worker_results: tuple[SwarmWorkerResult, ...]) -> SwarmCriticResult:
        received["task"] = task
        received["workers"] = worker_results
        return default_swarm_critic(task, worker_results)

    coordinator = _coordinator((RecordingAgent("chat", "ok"),), critic=critic)
    task = _task((_spec("w1", "chat"),), timeout_seconds=15)
    result = coordinator.execute(task)

    assert received["task"] is task
    assert len(received["workers"]) == 1  # type: ignore[index]
    assert result.critic_result is not None
    assert result.critic_result.status is SwarmCriticStatus.OK


# 8. Synthesizer recibe resultados + critic.
def test_synthesizer_receives_results_and_critic() -> None:
    received: dict[str, object] = {}

    def synthesizer(task, worker_results, critic_result) -> str:
        received["workers"] = worker_results
        received["critic"] = critic_result
        return "síntesis determinista"

    coordinator = _coordinator((RecordingAgent("chat", "ok"),), synthesizer=synthesizer)
    result = coordinator.execute(_task((_spec("w1", "chat"),), timeout_seconds=15))

    assert len(received["workers"]) == 1  # type: ignore[index]
    assert isinstance(received["critic"], SwarmCriticResult)  # type: ignore[index]
    assert result.final_synthesis == "síntesis determinista"


# 9. SwarmCoordinator no ejecuta tools críticas directamente.
def test_coordinator_cannot_execute_tools() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "ok"),))

    assert not hasattr(coordinator, "execute_tool")
    assert not hasattr(coordinator, "tool_executor")
    assert not hasattr(coordinator, "tool_registry")
    # Un worker cuyo agente no está registrado no ejecuta nada y falla de forma estructurada.
    result = coordinator.execute(_task((_spec("w1", "chat"), _spec("w2", "inexistente")), timeout_seconds=15))
    statuses = {worker.worker_id: worker.status for worker in result.worker_results}
    assert statuses["w2"] is SwarmWorkerStatus.AGENT_BLOCKED
    assert statuses["w1"] is SwarmWorkerStatus.SUCCEEDED


def test_default_allowlist_blocks_side_effect_agents() -> None:
    fake_coding = RecordingAgent("coding", "propongo cambio", fail=False)
    registry = AgentRegistry()
    registry.register(RecordingAgent("chat", "ok"))
    registry.register(fake_coding)
    coordinator = SwarmCoordinator(
        registry,
        _manager(),
        model_selection_policy=ModelSelectionPolicy(),
    )
    result = coordinator.execute(_task((_spec("w1", "coding"),), timeout_seconds=15))

    assert fake_coding.calls == []
    worker = result.worker_results[0]
    assert worker.status is SwarmWorkerStatus.AGENT_BLOCKED
    assert "allowlist" in worker.error


def test_allowlisted_agent_missing_from_registry_is_unavailable() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "ok"),))
    result = coordinator.execute(_task((_spec("w1", "project"),), timeout_seconds=15))

    worker = result.worker_results[0]
    assert worker.status is SwarmWorkerStatus.AGENT_UNAVAILABLE
    assert "not registered" in worker.error


# 10. No se modifica el comportamiento normal de una tarea que NO usa swarm.
def test_normal_orchestrator_behavior_without_swarm_is_untouched() -> None:
    orchestrator = _orchestrator(None)

    try:
        orchestrator.execute_swarm(_task((_spec("w1", "chat"),), timeout_seconds=15))
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert orchestrator._swarm_coordinator is None  # type: ignore[attr-defined]


def test_swarm_task_does_not_touch_conversation_memory() -> None:
    memory = ConversationMemory()
    orchestrator = _orchestrator(_coordinator((RecordingAgent("chat", "ok"),)))
    orchestrator._memory = memory  # type: ignore[attr-defined]

    orchestrator.execute_swarm(_task((_spec("w1", "chat"),), timeout_seconds=15))

    assert memory.history() == []


def test_default_synthesis_preserves_uncertainty_and_partial_failures() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "respuesta única"),))
    result = coordinator.execute(_task((_spec("w1", "chat"),), timeout_seconds=15))

    synthesis = result.final_synthesis
    assert synthesis is not None
    assert "Desacuerdos" not in synthesis
    assert "Fallos parciales" not in synthesis


def test_default_synthesis_reports_disagreements_without_inventing_consensus() -> None:
    agent_a = RecordingAgent("chat_a", "análisis A")
    agent_b = RecordingAgent("chat_b", "análisis B distinto")
    coordinator = _coordinator((agent_a, agent_b))
    result = coordinator.execute(_task((_spec("w1", "chat_a"), _spec("w2", "chat_b")), timeout_seconds=15))

    synthesis = result.final_synthesis
    assert synthesis is not None
    assert "Desacuerdos" in synthesis
    assert "no se inventa consenso" in synthesis
    assert result.critic_result is not None
    assert result.critic_result.disagreement_worker_ids == ("w1", "w2")


def test_invalid_task_is_rejected_without_executing_workers() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "ok"),))
    try:
        coordinator.execute(SwarmTask(task_id="t", objective="x", worker_specs=()))
        raised = False
    except InvalidSwarmTaskError:
        raised = True
    assert raised


def test_task_exceeding_max_workers_is_rejected_structurally() -> None:
    coordinator = _coordinator((RecordingAgent("chat", "ok"),), policy=SwarmPolicy(max_workers=1, worker_timeout_seconds=5))
    result = coordinator.execute(_task((_spec("w1", "chat"), _spec("w2", "chat")), timeout_seconds=15))
    assert result.status is SwarmStatus.INVALID_REQUEST
    assert result.error_code == "MAX_WORKERS_EXCEEDED"


def test_worker_runs_are_parallel_and_bounded() -> None:
    agent_a = RecordingAgent("chat_a", "a", delay=0.4)
    agent_b = RecordingAgent("chat_b", "b", delay=0.4)
    coordinator = _coordinator(
        (agent_a, agent_b),
        policy=SwarmPolicy(
            max_workers=2,
            worker_timeout_seconds=5,
            global_timeout_seconds=15,
            allowed_agent_names=DEFAULT_SWARM_AGENT_ALLOWLIST | frozenset({"chat_a", "chat_b"}),
        ),
    )
    started = time.monotonic()
    result = coordinator.execute(_task((_spec("w1", "chat_a"), _spec("w2", "chat_b")), timeout_seconds=15))
    elapsed = time.monotonic() - started

    assert result.status is SwarmStatus.SUCCESS
    assert elapsed < 0.9


# 11. Regresión E2E (smoke-1): el timeout efectivo del worker hereda el timeout
# explícito de la tarea; el default de política no lo trunca.
def test_worker_effective_timeout_inherits_task_timeout() -> None:
    agent = RecordingAgent("chat", "ok dentro del presupuesto de la tarea", delay=2.0)
    coordinator = _coordinator(
        (agent,),
        policy=SwarmPolicy(worker_timeout_seconds=1, global_timeout_seconds=30),
    )
    started = time.monotonic()
    result = coordinator.execute(_task((_spec("w1", "chat"),), timeout_seconds=5))
    elapsed = time.monotonic() - started

    worker = result.worker_results[0]
    assert worker.status is SwarmWorkerStatus.SUCCEEDED
    assert worker.duration_seconds >= 2.0
    assert result.status is SwarmStatus.SUCCESS
    assert elapsed < 5.0


# 12. Sin timeout de tarea explícito, el default de política sigue mandando.
def test_worker_effective_timeout_falls_back_to_policy_default() -> None:
    agent = RecordingAgent("chat", "nunca a tiempo", delay=3.0)
    coordinator = _coordinator(
        (agent,),
        policy=SwarmPolicy(worker_timeout_seconds=1, global_timeout_seconds=30),
    )
    result = coordinator.execute(_task((_spec("w1", "chat"),)))

    worker = result.worker_results[0]
    assert worker.status is SwarmWorkerStatus.TIMEOUT
    assert result.status is SwarmStatus.FAILED
