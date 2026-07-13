from pathlib import Path

from core.orchestrator import AtlasOrchestrator
from core.planner import Plan
from domain.refactoring.refactoring_plan import RefactoringPlan
from domain.refactoring.refactoring_result import RenameSymbolResult
from use_cases.refactoring_interaction import RefactoringInteractionUseCase


class _PlanRefactoringFake:
    def __init__(
        self,
        plan: RefactoringPlan,
    ) -> None:
        self.plan = plan

    def execute(
        self,
        symbol_name: str,
        new_name: str,
    ) -> RefactoringPlan:
        return RefactoringPlan(
            operation=self.plan.operation,
            symbol_name=symbol_name,
            new_name=new_name,
            definition_file=self.plan.definition_file,
            affected_files=self.plan.affected_files,
            risk_level=self.plan.risk_level,
            warnings=self.plan.warnings,
            can_apply=self.plan.can_apply,
            summary=self.plan.summary,
        )


class _RenameSymbolFake:
    def __init__(
        self,
        result: RenameSymbolResult,
    ) -> None:
        self.result = result
        self.calls: list[tuple[Path, str, str]] = []

    def execute(
        self,
        project_root: Path,
        symbol_name: str,
        new_name: str,
    ) -> RenameSymbolResult:
        self.calls.append((project_root, symbol_name, new_name))
        return self.result


def _plan(
    can_apply: bool = True,
    warnings: list[str] | None = None,
) -> RefactoringPlan:
    return RefactoringPlan(
        operation="rename_symbol",
        symbol_name="Router",
        new_name="AtlasRouter",
        definition_file="core/router.py",
        affected_files=[
            "bootstrap/bootstrap.py",
            "core/orchestrator.py",
            "core/router.py",
        ],
        risk_level="medium",
        warnings=warnings or [],
        can_apply=can_apply,
    )


def _result(
    applied: bool = True,
    rolled_back: bool = False,
    warnings: tuple[str, ...] = tuple(),
) -> RenameSymbolResult:
    return RenameSymbolResult(
        symbol_name="Router",
        new_name="AtlasRouter",
        changed_files=(Path("core/router.py"), Path("core/orchestrator.py")),
        replacements_count=5,
        applied=applied,
        rolled_back=rolled_back,
        warnings=warnings,
    )


def _use_case(
    plan: RefactoringPlan | None = None,
    result: RenameSymbolResult | None = None,
) -> tuple[RefactoringInteractionUseCase, _RenameSymbolFake]:
    rename = _RenameSymbolFake(result or _result())
    return (
        RefactoringInteractionUseCase(
            _PlanRefactoringFake(plan or _plan()),
            rename,
        ),
        rename,
    )


def test_parse_renombra_command() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("renombra Router a AtlasRouter")

    assert error is None
    assert command is not None
    assert command.symbol_name == "Router"
    assert command.new_name == "AtlasRouter"


def test_parse_renombrar_command() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("renombrar Router a AtlasRouter")

    assert error is None
    assert command is not None
    assert command.symbol_name == "Router"
    assert command.new_name == "AtlasRouter"


def test_parse_cambia_command() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("cambia Router por AtlasRouter")

    assert error is None
    assert command is not None
    assert command.symbol_name == "Router"
    assert command.new_name == "AtlasRouter"


def test_parse_rename_command() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("rename Router to AtlasRouter")

    assert error is None
    assert command is not None
    assert command.symbol_name == "Router"
    assert command.new_name == "AtlasRouter"


def test_parser_preserves_name_case() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("  ReNoMbRa   HTTPRouter   a   XMLRouter  ")

    assert error is None
    assert command is not None
    assert command.symbol_name == "HTTPRouter"
    assert command.new_name == "XMLRouter"


def test_rejects_incomplete_command() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("renombra Router")

    assert command is None
    assert error == "Orden de renombrado incompleta o no válida."


def test_rejects_qualified_expression() -> None:
    use_case, _ = _use_case()

    command, error = use_case.parse("renombra Planner.create_plan a create_task_plan")

    assert command is None
    assert error == "No se admiten expresiones cualificadas para este hito."


def test_shows_plan_before_apply(tmp_path: Path) -> None:
    use_case, _ = _use_case()
    prompts: list[str] = []

    response = use_case.execute(
        "renombra Router a AtlasRouter",
        tmp_path,
        lambda prompt: prompts.append(prompt) or "s",
    )

    assert prompts
    assert prompts[0].startswith("Plan de refactorización")
    assert "¿Deseas aplicar los cambios? [s/N]:" in prompts[0]
    assert response is not None
    assert response.startswith("Plan de refactorización")


def test_applies_when_confirmed_with_s(tmp_path: Path) -> None:
    use_case, rename = _use_case()

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "s")

    assert rename.calls == [(tmp_path, "Router", "AtlasRouter")]
    assert response is not None
    assert "Refactorización aplicada correctamente." in response


def test_applies_when_confirmed_with_si_accent(tmp_path: Path) -> None:
    use_case, rename = _use_case()

    use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "sí")

    assert rename.calls == [(tmp_path, "Router", "AtlasRouter")]


def test_applies_when_confirmed_with_yes(tmp_path: Path) -> None:
    use_case, rename = _use_case()

    use_case.execute("rename Router to AtlasRouter", tmp_path, lambda _: "yes")

    assert rename.calls == [(tmp_path, "Router", "AtlasRouter")]


def test_cancels_with_empty_response(tmp_path: Path) -> None:
    use_case, rename = _use_case()

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "")

    assert rename.calls == []
    assert response is not None
    assert "Refactorización cancelada" in response


def test_cancels_with_n(tmp_path: Path) -> None:
    use_case, rename = _use_case()

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "n")

    assert rename.calls == []
    assert response is not None
    assert "Refactorización cancelada" in response


def test_does_not_execute_rename_when_cancelled(tmp_path: Path) -> None:
    use_case, rename = _use_case()

    use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "no")

    assert rename.calls == []


def test_result_shows_changed_files(tmp_path: Path) -> None:
    use_case, _ = _use_case()

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "s")

    assert response is not None
    assert "- core/router.py" in response
    assert "- core/orchestrator.py" in response


def test_result_shows_replacements_count(tmp_path: Path) -> None:
    use_case, _ = _use_case()

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "s")

    assert response is not None
    assert "Reemplazos realizados: 5" in response


def test_handles_non_applicable_plan(tmp_path: Path) -> None:
    use_case, rename = _use_case(
        plan=_plan(can_apply=False, warnings=["símbolo inexistente: Router"])
    )

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "s")

    assert rename.calls == []
    assert response is not None
    assert "No se puede aplicar la refactorización." in response
    assert "símbolo inexistente: Router" in response


def test_handles_missing_symbol(tmp_path: Path) -> None:
    use_case, _ = _use_case(
        plan=RefactoringPlan(
            operation="rename_symbol",
            symbol_name="Missing",
            new_name="AtlasRouter",
            definition_file=None,
            affected_files=[],
            risk_level="high",
            warnings=["símbolo inexistente: Missing"],
            can_apply=False,
        )
    )

    response = use_case.execute("renombra Missing a AtlasRouter", tmp_path, lambda _: "s")

    assert response is not None
    assert "símbolo inexistente: Missing" in response
    assert "No se puede aplicar la refactorización." in response


def test_handles_rollback_without_traceback(tmp_path: Path) -> None:
    use_case, _ = _use_case(
        result=_result(
            applied=False,
            rolled_back=True,
            warnings=("rollback ejecutado: fallo controlado",),
        )
    )

    response = use_case.execute("renombra Router a AtlasRouter", tmp_path, lambda _: "s")

    assert response is not None
    assert "Rollback ejecutado." in response
    assert "Traceback" not in response


def test_unrelated_prompt_is_not_handled() -> None:
    use_case, _ = _use_case()

    response = use_case.execute(
        "analiza router.py",
        Path("."),
        lambda _: "s",
    )

    assert response is None


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
        return "respuesta previa"


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


class _RefactoringInteractionFake:
    def __init__(
        self,
        response: str | None,
    ) -> None:
        self.response = response
        self.calls: list[str] = []

    def execute(
        self,
        prompt: str,
        project_root: Path,
        confirm,
    ) -> str | None:
        self.calls.append(prompt)
        return self.response


def test_orchestrator_handles_refactoring_before_agent(
    monkeypatch,
    capsys,
) -> None:
    agent = _AgentFake()
    refactoring = _RefactoringInteractionFake("Plan de refactorización")
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_WriteFileFake(),
        refactoring_interaction=refactoring,
        project_root=Path("."),
    )
    prompts = iter(["renombra Router a AtlasRouter", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "Plan de refactorización" in output
    assert agent.calls == 0


def test_orchestrator_keeps_previous_flow_when_not_refactoring(
    monkeypatch,
    capsys,
) -> None:
    agent = _AgentFake()
    refactoring = _RefactoringInteractionFake(None)
    orchestrator = AtlasOrchestrator(
        planner=_PlannerFake(),
        router=_RouterFake(),
        model_manager=_ModelManagerFake(),
        memory=_MemoryFake(),
        registry=_RegistryFake(agent),
        write_file=_WriteFileFake(),
        refactoring_interaction=refactoring,
        project_root=Path("."),
    )
    prompts = iter(["hola", "salir"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))

    orchestrator.start()

    output = capsys.readouterr().out
    assert "respuesta previa" in output
    assert agent.calls == 1
