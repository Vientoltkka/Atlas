from __future__ import annotations

from typing import Any

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import (
    ArgumentField,
    ArgumentSchema,
    ArgumentSchemaRegistry,
    ArgumentValidator,
)
from tools.base_tool import BaseTool
from tools.execution_coordinator import (
    ExecutionCoordinationStatus,
    ExecutionCoordinator,
)
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.executor import ToolExecutor
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.single_tool_runner import SingleToolRunner
from tools.tool_chain_proposal_builder import ToolChainProposalBuilder
from tools.tool_chain_runner import ToolChainRunner
from tools.tool_context import ToolContext
from tools.tool_proposal_builder import ToolProposalBuilder


class FixedDecisionEngine:
    def __init__(self, decision: ExecutionDecision) -> None:
        self._decision = decision

    def decide(self, prompt: str) -> ExecutionDecision:
        return self._decision


class CountingReadTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Counting read."

    def execute(self, context: ToolContext) -> str:
        self.calls += 1
        return "alpha"


class CountingWriteTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0
        self.contents: list[str] = []

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Counting write."

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, context: ToolContext) -> str:
        self.calls += 1
        self.contents.append(str(context.parameters.get("content")))
        return "written"


def _coordinator() -> ExecutionCoordinator:
    return Bootstrap.build_execution_coordinator()


def _fake_chain_coordinator() -> tuple[ExecutionCoordinator, CountingReadTool, CountingWriteTool]:
    read = CountingReadTool()
    write = CountingWriteTool()
    registry = ToolRegistry()
    registry.register(read)
    registry.register(write)

    intent_registry = ToolIntentRegistry()
    intent_registry.register("file.read", "read_file")
    intent_registry.register("file.write", "write_file")
    selector = ToolSelector(registry, intent_registry)

    schema_registry = ArgumentSchemaRegistry()
    schema_registry.register(
        ArgumentSchema("file.read", (ArgumentField("path", str, required=True),))
    )
    schema_registry.register(
        ArgumentSchema(
            "file.write",
            (
                ArgumentField("path", str, required=True),
                ArgumentField("content", str, required=True),
            ),
        )
    )
    validator = ArgumentValidator(schema_registry)
    executor = ToolExecutor(registry)
    single = SingleToolRunner(selector, validator, executor)
    proposal_builder = ToolProposalBuilder(registry, selector, schema_registry, validator)
    chain_proposal_builder = ToolChainProposalBuilder(
        proposal_builder,
        selector,
        validator,
    )

    coordinator = ExecutionCoordinator(
        FixedDecisionEngine(
            ExecutionDecision(
                mode=ExecutionMode.TOOL_CHAIN,
                reason="fixed chain",
                confidence=0.9,
                candidate_tools=("file.read", "file.write"),
            )
        ),
        proposal_builder,
        chain_proposal_builder,
        single,
        ToolChainRunner(single),
    )
    return coordinator, read, write


def test_direct_response_request_does_not_execute_tools() -> None:
    result = _coordinator().execute("Hola, como estas?")

    assert result.status == ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED
    assert result.mode == ExecutionMode.DIRECT_RESPONSE
    assert result.executed is False
    assert result.execution_result is None


def test_knowledge_request_uses_direct_response_without_tools() -> None:
    result = _coordinator().execute("Explicame Clean Architecture")

    assert result.status == ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED
    assert result.executed is False


def test_single_read_executes_through_single_runner() -> None:
    result = _coordinator().execute("Lee el archivo README.md")

    assert result.status == ExecutionCoordinationStatus.EXECUTED
    assert result.mode == ExecutionMode.SINGLE_TOOL
    assert result.executed is True
    assert result.execution_result is not None
    assert result.execution_result.status == "success"


def test_single_missing_information_does_not_execute() -> None:
    result = _coordinator().execute("Lee este archivo")

    assert result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert result.missing_information == ("path",)
    assert result.executed is False
    assert result.execution_result is None


def test_single_ambiguous_request_does_not_execute() -> None:
    result = _coordinator().execute("Escribe algo en un archivo")

    assert result.status == ExecutionCoordinationStatus.AMBIGUOUS_REQUEST
    assert set(result.ambiguous_information) == {"path", "content"}
    assert result.executed is False


def test_delete_request_is_safe_direct_or_unsupported_without_execution() -> None:
    result = _coordinator().execute("Borra README.md")

    assert result.status == ExecutionCoordinationStatus.UNSUPPORTED
    assert result.executed is False
    assert result.execution_result is None


def test_write_requires_confirmation_and_does_not_create_file_before_confirm(tmp_path) -> None:
    target = tmp_path / "prueba.txt"
    coordinator = _coordinator()

    result = coordinator.execute(f"Escribe hola en {target}")

    assert result.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert result.confirmation_id
    assert result.executed is False
    assert target.exists() is False


def test_write_confirmation_yes_executes_once(tmp_path) -> None:
    target = tmp_path / "prueba.txt"
    coordinator = _coordinator()
    pending = coordinator.execute(f"Escribe hola en {target}")

    confirmed = coordinator.confirm(str(pending.confirmation_id), "si")
    repeated = coordinator.confirm(str(pending.confirmation_id), "si")

    assert confirmed.status == ExecutionCoordinationStatus.EXECUTED
    assert target.read_text(encoding="utf-8") == "hola"
    assert repeated.status == ExecutionCoordinationStatus.FAILED
    assert target.read_text(encoding="utf-8") == "hola"


def test_write_confirmation_no_does_not_create_file(tmp_path) -> None:
    target = tmp_path / "prueba.txt"
    coordinator = _coordinator()
    pending = coordinator.execute(f"Escribe hola en {target}")

    cancelled = coordinator.confirm(str(pending.confirmation_id), "no")

    assert cancelled.status == ExecutionCoordinationStatus.CANCELLED
    assert target.exists() is False


def test_chain_pauses_before_dangerous_write_and_resumes_without_repeating_read() -> None:
    coordinator, read, write = _fake_chain_coordinator()
    pending = coordinator.execute("Lee README.md y copia su contenido en out.txt")

    assert pending.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert pending.executed is True
    assert read.calls == 1
    assert write.calls == 0

    confirmed = coordinator.confirm(str(pending.confirmation_id), "s")
    repeated = coordinator.confirm(str(pending.confirmation_id), "s")

    assert confirmed.status == ExecutionCoordinationStatus.EXECUTED
    assert read.calls == 1
    assert write.calls == 1
    assert write.contents == ["alpha"]
    assert repeated.status == ExecutionCoordinationStatus.FAILED
    assert read.calls == 1
    assert write.calls == 1


def test_chain_missing_information_does_not_execute() -> None:
    result = _coordinator().execute("Lee este archivo y guardalo en otro")

    assert result.status in {
        ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED,
        ExecutionCoordinationStatus.INFORMATION_REQUIRED,
    }
    assert result.executed is False


def test_forced_chain_with_unregistered_tool_does_not_execute_partial_steps() -> None:
    coordinator = ExecutionCoordinator(
        FixedDecisionEngine(
            ExecutionDecision(
                mode=ExecutionMode.TOOL_CHAIN,
                reason="forced unsupported",
                confidence=0.9,
                candidate_tools=("file.delete", "file.write"),
            )
        ),
        Bootstrap.build_tool_proposal_builder(),
        Bootstrap.build_tool_chain_proposal_builder(),
        Bootstrap.build_single_tool_runner(),
        Bootstrap.build_tool_chain_runner(),
    )

    result = coordinator.execute("Borra README.md y crea otro")

    assert result.status == ExecutionCoordinationStatus.UNSUPPORTED
    assert result.executed is False
    assert result.execution_result is None


def test_wrong_confirmation_id_does_not_execute() -> None:
    result = _coordinator().confirm("wrong-id", "s")

    assert result.status == ExecutionCoordinationStatus.FAILED
    assert result.executed is False


def test_ambiguous_confirmation_keeps_pending_operation(tmp_path) -> None:
    target = tmp_path / "prueba.txt"
    coordinator = _coordinator()
    pending = coordinator.execute(f"Escribe hola en {target}")

    ambiguous = coordinator.confirm(str(pending.confirmation_id), "quizas")

    assert ambiguous.status == ExecutionCoordinationStatus.CONFIRMATION_REQUIRED
    assert target.exists() is False
    confirmed = coordinator.confirm(str(pending.confirmation_id), "s")

    assert confirmed.status == ExecutionCoordinationStatus.EXECUTED
    assert target.read_text(encoding="utf-8") == "hola"


def test_confirmation_id_from_other_coordinator_does_not_execute(tmp_path) -> None:
    target = tmp_path / "prueba.txt"
    owner = _coordinator()
    other = _coordinator()
    pending = owner.execute(f"Escribe hola en {target}")

    wrong_owner = other.confirm(str(pending.confirmation_id), "s")

    assert wrong_owner.status == ExecutionCoordinationStatus.FAILED
    assert target.exists() is False


def test_empty_input_returns_safe_result() -> None:
    result = _coordinator().execute("   ")

    assert result.status == ExecutionCoordinationStatus.DIRECT_RESPONSE_REQUIRED
    assert result.executed is False


def test_result_contains_mode_status_executed_and_message() -> None:
    result = _coordinator().execute("Lee este archivo")

    assert result.mode == ExecutionMode.SINGLE_TOOL
    assert result.status == ExecutionCoordinationStatus.INFORMATION_REQUIRED
    assert result.executed is False
    assert result.message


def test_coordinator_module_does_not_import_tool_executor_directly() -> None:
    import tools.execution_coordinator as module

    assert "ToolExecutor" not in module.__dict__
