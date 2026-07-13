from pathlib import Path

from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from use_cases.correction_interaction import (
    CorrectionInteractionUseCase,
    CorrectionTestResult,
)
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


class _WriteFileFake:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on_content: str | None = None

    def execute(
        self,
        path: str,
        content: str,
    ) -> str:
        self.calls.append((path, content))

        if self.fail_on_content is not None and content == self.fail_on_content:
            raise RuntimeError("fallo de escritura")

        Path(path).write_text(content, encoding="utf-8")
        return "ok"


class _QueryArchitectureFake:
    def __init__(
        self,
        affected_files: list[str] | None = None,
    ) -> None:
        self.dependencies_calls: list[str] = []
        self.impact_calls: list[str] = []
        self.affected_files = affected_files or [
            "bootstrap/bootstrap.py",
            "core/orchestrator.py",
        ]

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
            affected_files=self.affected_files,
        )


class _PromptClientFake:
    def __init__(
        self,
        proposed_file: str = "class Router:\n    pass",
    ) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []
        self.proposed_file = proposed_file
        self.raw_response: str | None = None

    def ask(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        self.calls.append((model, messages))

        if self.raw_response is not None:
            return self.raw_response

        return "\n".join(
            [
                "PROBLEM:",
                "condición duplicada.",
                "",
                "RISK:",
                "low",
                "",
                "PROPOSED_FILE:",
                "```python",
                self.proposed_file,
                "```",
            ]
        )


class _ConfirmFake:
    def __init__(
        self,
        answer: str,
    ) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(
        self,
        prompt: str,
    ) -> str:
        self.prompts.append(prompt)
        return self.answer


def _write(
    path: Path,
    content: str = "class Router:\n    pass\n",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _use_case(
    content: str = "class Router:\n    pass\n",
    proposed_file: str = "class Router:\n    pass",
    affected_files: list[str] | None = None,
) -> tuple[
    CorrectionInteractionUseCase,
    _ReadFileFake,
    _WriteFileFake,
    _QueryArchitectureFake,
    _PromptClientFake,
]:
    read_file = _ReadFileFake(content)
    write_file = _WriteFileFake()
    query = _QueryArchitectureFake(affected_files)
    prompt = _PromptClientFake(proposed_file)
    use_case = CorrectionInteractionUseCase(
        read_file,
        write_file,
        query,
        prompt,
    )
    use_case._run_tests = lambda project_root: CorrectionTestResult(
        tests_run=104,
        tests_passed=104,
        tests_failed=0,
        success=True,
        output="104 passed",
    )
    return (use_case, read_file, write_file, query, prompt)


def _choose_model(task: str) -> str:
    assert task == "coding"
    return "test-coding-model"


def test_parse_supported_commands() -> None:
    use_case, _, _, _, _ = _use_case()

    for prompt in (
        "corrige core/router.py",
        "arregla core/router.py",
        "mejora core/router.py",
        "fix core/router.py",
    ):
        command, error = use_case.parse(prompt)

        assert error is None
        assert command is not None
        assert command.path == "core/router.py"


def test_parser_preserves_path() -> None:
    use_case, _, _, _, _ = _use_case()

    command, error = use_case.parse("  CoRrIgE   core/router.py  ")

    assert error is None
    assert command is not None
    assert command.path == "core/router.py"


def test_rejects_incomplete_command() -> None:
    use_case, _, _, _, _ = _use_case()

    command, error = use_case.parse("corrige")

    assert command is None
    assert error == "Orden de corrección incompleta o no válida."


def test_rejects_path_outside_project_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    _write(outside)
    use_case, _, _, _, _ = _use_case()

    response = use_case.execute(
        f"corrige {outside}",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert response is not None
    assert "archivo fuera de project_root" in response
    outside.unlink()


def test_rejects_missing_file(tmp_path: Path) -> None:
    use_case, _, _, _, _ = _use_case()

    response = use_case.execute(
        "corrige missing.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert response is not None
    assert "archivo inexistente: missing.py" in response


def test_rejects_non_python_file(tmp_path: Path) -> None:
    _write(tmp_path / "router.txt", "Router\n")
    use_case, _, _, _, _ = _use_case()

    response = use_case.execute(
        "corrige router.txt",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert response == "Solo se admiten archivos Python .py."


def test_reads_file_with_existing_capability(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, read_file, _, _, _ = _use_case()

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("n"),
    )

    assert read_file.calls == [str((tmp_path / "core" / "router.py").resolve())]


def test_queries_architectural_context_and_impact(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, query, _ = _use_case()

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("n"),
    )

    assert query.dependencies_calls == ["core/router.py"]
    assert query.impact_calls == ["core/router.py"]


def test_calls_model_with_file_content(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _, prompt = _use_case("class Router:\n    pass\n")

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("n"),
    )

    assert prompt.calls
    assert "class Router" in prompt.calls[0][1][1]["content"]


def test_prompt_requires_structured_full_file_output(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, _, _, prompt = _use_case()

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("n"),
    )

    full_prompt = "\n".join(message["content"] for message in prompt.calls[0][1])
    assert "No inventar módulos inexistentes." in full_prompt
    assert "No modificar otros archivos." in full_prompt
    assert "PROPOSED_FILE" in full_prompt
    assert "No devolver diff." in full_prompt


def test_rejects_ambiguous_free_text_proposal(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    use_case, _, write_file, _, prompt = _use_case()
    prompt.raw_response = "Cambia una condición."

    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert response is not None
    assert "PROBLEM, RISK y PROPOSED_FILE" in response
    assert write_file.calls == []


def test_response_shows_target_file_affected_files_risk_and_preview(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "core" / "router.py")
    confirm = _ConfirmFake("n")
    use_case, _, _, _, _ = _use_case()

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        confirm,
    )

    assert confirm.prompts
    assert "Archivo: core/router.py" in confirm.prompts[0]
    assert "- bootstrap/bootstrap.py" in confirm.prompts[0]
    assert "- core/orchestrator.py" in confirm.prompts[0]
    assert "Riesgo: low" in confirm.prompts[0]
    assert "Vista previa:" in confirm.prompts[0]
    assert "```python" in confirm.prompts[0]


def test_medium_risk_from_three_to_five_affected_files(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    confirm = _ConfirmFake("n")
    use_case, _, _, _, _ = _use_case(
        affected_files=[
            "a.py",
            "b.py",
            "c.py",
        ]
    )

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        confirm,
    )

    assert confirm.prompts
    assert "Riesgo: medium" in confirm.prompts[0]


def test_high_risk_from_more_than_five_affected_files(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    confirm = _ConfirmFake("n")
    use_case, _, _, _, _ = _use_case(
        affected_files=[
            "a.py",
            "b.py",
            "c.py",
            "d.py",
            "e.py",
            "f.py",
        ]
    )

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        confirm,
    )

    assert confirm.prompts
    assert "Riesgo: high" in confirm.prompts[0]


def test_confirmation_prompt_shows_plan_before_apply(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "router.py")
    confirm = _ConfirmFake("n")
    use_case, _, _, _, _ = _use_case()

    use_case.execute("corrige core/router.py", tmp_path, _choose_model, confirm)

    assert confirm.prompts
    assert "Propuesta de corrección" in confirm.prompts[0]
    assert "¿Deseas aplicar la corrección? [s/N]:" in confirm.prompts[0]


def test_cancel_with_empty_answer_does_not_modify_file_or_run_tests(
    tmp_path: Path,
) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file="class Router:\n    value = 1",
    )
    ran_tests = False

    def run_tests(project_root: Path) -> CorrectionTestResult:
        nonlocal ran_tests
        ran_tests = True
        return CorrectionTestResult(0, 0, 0, True, "")

    use_case._run_tests = run_tests
    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake(""),
    )

    assert target.read_text(encoding="utf-8") == original
    assert write_file.calls == []
    assert ran_tests is False
    assert response is not None
    assert "Corrección cancelada." in response


def test_cancel_with_n_does_not_modify_file(tmp_path: Path) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file="class Router:\n    value = 1",
    )

    use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("n"),
    )

    assert target.read_text(encoding="utf-8") == original
    assert write_file.calls == []


def test_apply_after_affirmative_confirmation_runs_syntax_and_tests(
    tmp_path: Path,
) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    proposed = "class Router:\n    value = 1"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file=proposed,
    )

    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert target.read_text(encoding="utf-8") == proposed
    assert write_file.calls == [(str(target.resolve()), proposed)]
    assert response is not None
    assert "Validación de sintaxis: correcta" in response
    assert "Tests ejecutados: 104" in response
    assert "Tests superados: 104" in response
    assert "Tests fallidos: 0" in response
    assert "Corrección aplicada correctamente." in response
    assert "Cambios aún sin commit." in response


def test_apply_accepts_spanish_affirmative_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    proposed = "class Router:\n    value = 1"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file=proposed,
    )

    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("sí"),
    )

    assert target.read_text(encoding="utf-8") == proposed
    assert write_file.calls == [(str(target.resolve()), proposed)]
    assert response is not None
    assert "Corrección aplicada correctamente." in response


def test_invalid_syntax_is_rejected_without_partial_change(tmp_path: Path) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file="class Router(",
    )

    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert target.read_text(encoding="utf-8") == original
    assert write_file.calls == []
    assert response is not None
    assert "La corrección no superó la validación." in response
    assert "El proyecto no quedó parcialmente modificado." in response
    assert "Traceback" not in response


def test_write_failure_rolls_back_to_original(tmp_path: Path) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    proposed = "class Router:\n    value = 1"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file=proposed,
    )
    write_file.fail_on_content = proposed

    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert target.read_text(encoding="utf-8") == original
    assert response is not None
    assert "Se restauró el contenido original." in response


def test_test_failure_rolls_back_to_original(tmp_path: Path) -> None:
    target = tmp_path / "core" / "router.py"
    original = "class Router:\n    pass\n"
    proposed = "class Router:\n    value = 1"
    _write(target, original)
    use_case, _, write_file, _, _ = _use_case(
        content=original,
        proposed_file=proposed,
    )
    use_case._run_tests = lambda project_root: CorrectionTestResult(
        tests_run=104,
        tests_passed=103,
        tests_failed=1,
        success=False,
        output="1 failed, 103 passed",
    )

    response = use_case.execute(
        "corrige core/router.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

    assert target.read_text(encoding="utf-8") == original
    assert write_file.calls == [
        (str(target.resolve()), proposed),
        (str(target.resolve()), original),
    ]
    assert response is not None
    assert "La corrección no superó la validación." in response
    assert "tests fallidos: 1" in response


def test_controlled_errors_do_not_show_traceback(tmp_path: Path) -> None:
    use_case, _, _, _, _ = _use_case()

    response = use_case.execute(
        "corrige missing.py",
        tmp_path,
        _choose_model,
        _ConfirmFake("s"),
    )

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


class _LegacyWriteFileFake:
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
        self.confirm_received = False

    def execute(
        self,
        prompt: str,
        project_root: Path,
        choose_model,
        confirm,
    ) -> str | None:
        self.confirm_received = confirm is not None
        return self.response


def test_orchestrator_handles_correction_before_agent(monkeypatch, capsys) -> None:
    agent = _AgentFake()
    correction = _CorrectionInteractionFake("Propuesta de corrección")
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_LegacyWriteFileFake(),
        correction_interaction=correction,
        project_root=Path("."),
    )
    prompts = iter(["corrige core/router.py", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "Propuesta de corrección" in output
    assert correction.confirm_received is True
    assert agent.calls == 0


def test_orchestrator_keeps_previous_commands_working(monkeypatch, capsys) -> None:
    agent = _AgentFake()
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_LegacyWriteFileFake(),
        correction_interaction=_CorrectionInteractionFake(None),
        project_root=Path("."),
    )
    prompts = iter(["hola", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "respuesta anterior" in output
    assert agent.calls == 1
