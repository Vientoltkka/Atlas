from pathlib import Path

from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from use_cases.correction_interaction import CorrectionInteractionUseCase
from use_cases.query_architecture_graph import ArchitectureQueryResult


class _ReadFileFake:
    def __init__(
        self,
        content: str = "class Router:\n    pass\n",
    ) -> None:
        self.content = content
        self.calls: list[str] = []

    def execute(
        self,
        path: str,
    ) -> str:
        self.calls.append(path)
        return self.content


class _QueryArchitectureFake:
    def __init__(self) -> None:
        self.dependencies_calls: list[str] = []
        self.impact_calls: list[str] = []

    def dependencies_of(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        self.dependencies_calls.append(target)
        return ArchitectureQueryResult(
            target=target,
            dependencies=["core/planner.py"],
        )

    def impact_of(
        self,
        target: str,
    ) -> ArchitectureQueryResult:
        self.impact_calls.append(target)
        return ArchitectureQueryResult(
            target=target,
            affected_files=[
                "bootstrap/bootstrap.py",
                "core/orchestrator.py",
            ],
        )


class _PromptClientFake:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def ask(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        self.calls.append((model, messages))
        return "\n".join(
            [
                "Problema: condición duplicada.",
                "Solución: simplificar la rama.",
                "```python",
                "class Router:",
                "    pass",
                "```",
            ]
        )


def _write(
    path: Path,
    content: str = "class Router:\n    pass\n",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _use_case(
    content: str = "class Router:\n    pass\n",
) -> tuple[
    CorrectionInteractionUseCase,
    _ReadFileFake,
    _QueryArchitectureFake,
    _PromptClientFake,
]:
    read_file = _ReadFileFake(content)
    query = _QueryArchitectureFake()
    prompt = _PromptClientFake()
    return (
        CorrectionInteractionUseCase(read_file, query, prompt),
        read_file,
        query,
        prompt,
    )


def _choose_model(task: str) -> str:
    assert task == "coding"
    return "test-coding-model"


def test_parse_corrige_command() -> None:
    use_case, _, _, _ = _use_case()

    command, error = use_case.parse("corrige core/router.py")

    assert error is None
    assert command is not None
    assert command.path == "core/router.py"


def test_parse_arregla_command() -> None:
    use_case, _, _, _ = _use_case()

    command, error = use_case.parse("arregla core/router.py")

    assert error is None
    assert command is not None
    assert command.path == "core/router.py"


def test_parse_mejora_command() -> None:
    use_case, _, _, _ = _use_case()

    command, error = use_case.parse("mejora core/router.py")

    assert error is None
    assert command is not None
    assert command.path == "core/router.py"


def test_parse_fix_command() -> None:
    use_case, _, _, _ = _use_case()

    command, error = use_case.parse("fix core/router.py")

    assert error is None
    assert command is not None
    assert command.path == "core/router.py"


def test_parser_preserves_path() -> None:
    use_case, _, _, _ = _use_case()

    command, error = use_case.parse("  CoRrIgE   core/router.py  ")

    assert error is None
    assert command is not None
    assert command.path == "core/router.py"


def test_rejects_incomplete_command() -> None:
    use_case, _, _, _ = _use_case()

    command, error = use_case.parse("corrige")

    assert command is None
    assert error == "Orden de corrección incompleta o no válida."


def test_rejects_path_outside_project_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    _write(outside)
    use_case, _, _, _ = _use_case()

    response = use_case.execute(
        f"corrige {outside}",
        tmp_path,
        _choose_model,
    )

    assert response is not None
    assert "archivo fuera de project_root" in response
    outside.unlink()


def test_rejects_missing_file(tmp_path: Path) -> None:
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige missing.py", tmp_path, _choose_model)

    assert response is not None
    assert "archivo inexistente: missing.py" in response


def test_rejects_non_python_file(tmp_path: Path) -> None:
    _write(tmp_path / "router.txt", "Router\n")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige router.txt", tmp_path, _choose_model)

    assert response == "Solo se admiten archivos Python .py."


def test_reads_file_with_existing_capability(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, read_file, _, _ = _use_case()

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert read_file.calls == [str((tmp_path / "core" / "router.py").resolve())]


def test_queries_architectural_context(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, query, _ = _use_case()

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert query.dependencies_calls == ["core/router.py"]


def test_queries_impact(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, query, _ = _use_case()

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert query.impact_calls == ["core/router.py"]


def test_calls_model_with_file_content(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, prompt = _use_case("class Router:\n    pass\n")

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert prompt.calls
    assert "class Router" in prompt.calls[0][1][1]["content"]


def test_prompt_includes_clear_restrictions(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, prompt = _use_case()

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    full_prompt = "\n".join(message["content"] for message in prompt.calls[0][1])
    assert "No inventar módulos inexistentes." in full_prompt
    assert "No modificar otros archivos." in full_prompt
    assert "Devolver código Python completo o diff estructurado." in full_prompt


def test_returns_concrete_proposal(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert response is not None
    assert "Problema: condición duplicada." in response
    assert "class Router:" in response


def test_response_shows_target_file(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert response is not None
    assert "Archivo:\ncore/router.py" in response


def test_response_shows_affected_files(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert response is not None
    assert "- bootstrap/bootstrap.py" in response
    assert "- core/orchestrator.py" in response


def test_response_shows_risk(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert response is not None
    assert "Riesgo:\nlow" in response


def test_response_shows_code_or_diff_preview(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert response is not None
    assert "Vista previa:" in response
    assert "```python" in response


def test_does_not_modify_file(tmp_path: Path) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    _write(target, original)
    use_case, _, _, _ = _use_case("class Router:\n    pass\n")

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert target.read_text(encoding="utf-8") == original


def test_does_not_execute_tests(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert response is not None
    assert "pytest" not in response


def test_does_not_execute_commands(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, prompt = _use_case()

    use_case.execute("corrige core/router.py", tmp_path, _choose_model)

    assert len(prompt.calls) == 1


def test_controlled_errors_do_not_show_traceback(tmp_path: Path) -> None:
    use_case, _, _, _ = _use_case()

    response = use_case.execute("corrige missing.py", tmp_path, _choose_model)

    assert response is not None
    assert "Traceback" not in response


class _PlannerFake:
    def create_plan(
        self,
        prompt: str,
    ) -> Plan:
        return Plan(task="chat", objective=prompt)


class _RouterFake:
    def route(
        self,
        plan: Plan,
    ) -> str:
        return "chat"


class _ModelManagerFake:
    def choose_model(
        self,
        task: str,
    ) -> str:
        return "test-model"


class _MemoryFake:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add_user(
        self,
        prompt: str,
    ) -> None:
        self.messages.append({"role": "user", "content": prompt})

    def add_assistant(
        self,
        response: str,
    ) -> None:
        self.messages.append({"role": "assistant", "content": response})

    def history(self) -> list[dict[str, str]]:
        return self.messages


class _AgentFake:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        self.calls += 1
        return "respuesta anterior"


class _RegistryFake:
    def __init__(
        self,
        agent: _AgentFake,
    ) -> None:
        self._agent = agent

    def get(
        self,
        name: str,
    ):
        if name == "chat":
            return self._agent

        return None


class _WriteFileFake:
    def execute(
        self,
        path: str,
        content: str,
    ) -> str:
        return "ok"


class _CorrectionInteractionFake:
    def __init__(
        self,
        response: str | None,
    ) -> None:
        self.response = response

    def execute(
        self,
        prompt: str,
        project_root: Path,
        choose_model,
    ) -> str | None:
        return self.response


def test_orchestrator_handles_correction_before_agent(monkeypatch, capsys) -> None:
    agent = _AgentFake()
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_WriteFileFake(),
        correction_interaction=_CorrectionInteractionFake("Propuesta de corrección"),
        project_root=Path("."),
    )
    prompts = iter(["corrige core/router.py", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "Propuesta de corrección" in output
    assert agent.calls == 0


def test_orchestrator_keeps_previous_commands_working(monkeypatch, capsys) -> None:
    agent = _AgentFake()
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_WriteFileFake(),
        correction_interaction=_CorrectionInteractionFake(None),
        project_root=Path("."),
    )
    prompts = iter(["hola", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "respuesta anterior" in output
    assert agent.calls == 1
