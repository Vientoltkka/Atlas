from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from services.pdf_service import PdfService
from tools.documents.create_training_pdf_tool import CreateTrainingPdfTool
from tools.execution_coordinator import ExecutionCoordinator
from tools.execution_decision import ExecutionDecisionEngine
from tools.executor import ToolExecutor
from tools.intent_selector import ToolIntentRegistry, ToolSelector
from tools.registry import ToolRegistry
from tools.single_tool_runner import SingleToolRunner
from tools.tool_context import ToolContext
from tools.tool_proposal_builder import ToolProposalBuilder
from tools.tool_chain_proposal_builder import ToolChainProposalBuilder
from use_cases.create_training_pdf import CreateTrainingPdfUseCase
from use_cases.execution_conversation import ExecutionConversationController


REALISTIC_TRAINING_CONTENT = """# Entrenamiento CrossFit para ma\u00f1ana

**Objetivo:** potencia y acondicionamiento.
- Calentamiento: movilidad de cadera y hombro.
- T\u00e9cnica: sentadilla frontal con \u00e9nfasis en posici\u00f3n.
---
AMRAP 18: 12/10 calor\u00edas, 10 burpees por encima de la barra y 8 sentadillas frontales con carga escalable; mant\u00e9n una respiraci\u00f3n controlada durante todo el bloque para conservar la t\u00e9cnica.
X""" + "X" * 1_000 + "\n\nPor motivos de seguridad y privacidad, no puedo crear archivos PDF desde aquí."

class OpenSpy:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.error = error

    def execute(self, context: ToolContext) -> str:
        path = str(context.parameters["path"])
        self.calls.append(path)
        if self.error:
            raise self.error
        return path


class BrokenPdf:
    def create(self, content: str, target: Path) -> None:
        raise RuntimeError("renderer down")


def _use_case(opener: OpenSpy, service=None) -> CreateTrainingPdfUseCase:
    return CreateTrainingPdfUseCase(
        service or PdfService(), opener, now_provider=lambda: datetime(2026, 8, 30, 10, 0, 0)
    )


def test_pdf_is_valid_and_spanish_text_creates_output_directory(tmp_path: Path) -> None:
    opener = OpenSpy()
    result = _use_case(opener).execute("Sesión para mañana: técnica y calentamiento.", tmp_path / "artifacts" / "documents")
    path = Path(result.removeprefix("PDF creado y abierto: "))
    assert path.read_bytes().startswith(b"%PDF-")
    assert path.parent.exists()
    assert opener.calls == [str(path)]


def test_pdf_renders_realistic_training_markdown_without_horizontal_space_error(
    tmp_path: Path,
) -> None:
    opener = OpenSpy()
    result = _use_case(opener).execute(REALISTIC_TRAINING_CONTENT, tmp_path)
    path = Path(result.removeprefix("PDF creado y abierto: "))
    assert path.read_bytes().startswith(b"%PDF-")
    assert path.stat().st_size > 1_000
    assert opener.calls == [str(path)]


def test_pdf_export_lines_keep_training_text_and_remove_pdf_refusal() -> None:
    lines = PdfService._export_lines(REALISTIC_TRAINING_CONTENT)
    rendered = "\n".join(lines)
    assert "**Objetivo:**" in rendered
    assert "- Calentamiento:" in rendered
    assert "---" in rendered
    assert "ma\u00f1ana" in rendered
    assert "no puedo crear archivos pdf" not in rendered.casefold()

def test_pdf_long_content_paginates(tmp_path: Path) -> None:
    opener = OpenSpy()
    result = _use_case(opener).execute(
        ("Bloque de entrenamiento y recuperación.\n" * 300),
        tmp_path,
    )
    path = Path(result.removeprefix("PDF creado y abierto: "))
    assert path.read_bytes().count(b"/Type /Page") >= 3
    assert opener.calls == [str(path)]

def test_empty_content_is_rejected_without_opening(tmp_path: Path) -> None:
    opener = OpenSpy()
    with pytest.raises(ValueError, match="vacio"):
        _use_case(opener).execute("   ", tmp_path)
    assert opener.calls == []


def test_pdf_names_do_not_collide(tmp_path: Path) -> None:
    opener = OpenSpy()
    use_case = _use_case(opener)
    first = use_case.execute("uno", tmp_path)
    second = use_case.execute("dos", tmp_path)
    assert first != second
    assert len(list(tmp_path.glob("*.pdf"))) == 2


def test_pdf_creation_failure_is_observable_and_does_not_open(tmp_path: Path) -> None:
    opener = OpenSpy()
    with pytest.raises(RuntimeError, match="renderer down"):
        _use_case(opener, BrokenPdf()).execute("plan", tmp_path)
    assert opener.calls == []


def test_open_failure_keeps_created_pdf_observable(tmp_path: Path) -> None:
    opener = OpenSpy(RuntimeError("viewer down"))
    with pytest.raises(RuntimeError, match="PDF creado en .*viewer down"):
        _use_case(opener).execute("plan", tmp_path)
    assert len(list(tmp_path.glob("*.pdf"))) == 1


def test_confirmation_yes_creates_once_and_no_cancels(tmp_path: Path) -> None:
    opener = OpenSpy()
    registry = ToolRegistry()
    registry.register(CreateTrainingPdfTool(_use_case(opener)))
    intents = ToolIntentRegistry()
    intents.register("training.pdf.create", "training.create_pdf")
    runner = SingleToolRunner(ToolSelector(registry, intents), _validator(), ToolExecutor(registry))
    controller = ExecutionConversationController(_coordinator(runner))

    pending = controller.handle_registered_tool("training.create_pdf", {"content": "plan", "output_dir": str(tmp_path)}, original_text="crea PDF", confirmation_text="Voy a crear y abrir el PDF. ¿Confirmas?")
    assert pending.text == "Voy a crear y abrir el PDF. ¿Confirmas?"
    assert not list(tmp_path.glob("*.pdf"))
    done = controller.handle("sí")
    assert done.result.executed is True
    assert len(list(tmp_path.glob("*.pdf"))) == 1
    assert len(opener.calls) == 1

    pending = controller.handle_registered_tool("training.create_pdf", {"content": "plan", "output_dir": str(tmp_path / "cancel")}, original_text="crea PDF", confirmation_text="Voy a crear y abrir el PDF. ¿Confirmas?")
    assert pending.result.confirmation_id
    controller.handle("cancelar")
    assert not (tmp_path / "cancel").exists()
    assert len(opener.calls) == 1


def _validator():
    from tools.argument_schema import ArgumentSchema, ArgumentSchemaRegistry, ArgumentValidator, ArgumentField
    schemas = ArgumentSchemaRegistry()
    schemas.register(ArgumentSchema("training.pdf.create", (ArgumentField("content", str, required=True), ArgumentField("output_dir", str, required=True))))
    return ArgumentValidator(schemas)


def _coordinator(runner: SingleToolRunner) -> ExecutionCoordinator:
    from tools.execution_decision import ExecutionDecisionEngine
    return ExecutionCoordinator(ExecutionDecisionEngine(("training.pdf.create",)), None, None, runner, None)  # type: ignore[arg-type]

def test_training_agent_pdf_request_stays_pending_then_confirms_once(tmp_path: Path, monkeypatch) -> None:
    from bootstrap.bootstrap import Bootstrap

    orchestrator = Bootstrap.build()
    training = orchestrator._registry.get("training")
    calls: list[object] = []

    def generate(*, model, messages):
        calls.append((model, messages))
        return "Entrenamiento CrossFit de 60 minutos para mañana."

    monkeypatch.setattr(training, "run", generate)
    orchestrator._training_pdf_output_dir = tmp_path
    tool = orchestrator._tool_registry.get("training.create_pdf")
    opener = OpenSpy()
    tool._use_case._open_file_tool = opener

    pending = orchestrator.process_prompt(
        "Créame un entrenamiento de CrossFit de 60 minutos para mañana y guárdalo en PDF",
        confirm=lambda _prompt: "",
    )
    assert "Voy a crear y abrir el PDF. ¿Confirmas?" in pending
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.pdf"))

    confirmed = orchestrator.process_prompt("sí", confirm=lambda _prompt: "")
    assert "PDF creado y abierto:" in confirmed
    assert len(calls) == 1
    assert len(list(tmp_path.glob("*.pdf"))) == 1
    assert len(opener.calls) == 1