from __future__ import annotations

from pathlib import Path
import shutil

from core.dev_worker import DevStep, DevWorkerAction
from core.dev_worker_route import (
    CONTROL_FILE_RELATIVE,
    FOCAL_TEST_RELATIVE,
    DevWorkerRoute,
)
from core.orchestrator import AtlasOrchestrator

_ROOT = Path(__file__).resolve().parents[1]
_E2E_PROMPT = (
    "Atlas, revisa el archivo de prueba controlado del Computer Worker, "
    "corrige el valor incorrecto y ejecuta su test focal para verificarlo."
)
_STT_VARIANT_PROMPT = (
    "Atlas, revisa el archivo de prueba controlado del Computeer World "
    "y verifica que su test focal funciona."
)


def _tmp_project(tmp_path: Path, *, correct: bool) -> Path:
    project = tmp_path / "project"
    (project / "tests" / "fixtures").mkdir(parents=True)
    control = project / CONTROL_FILE_RELATIVE
    control.write_text(
        '"""Controlled Computer Worker fixture."""\n\nCONTROL_VALUE = "%s"\n'
        % ("valor-correcto" if correct else "valor-incorrecto"),
        encoding="utf-8",
    )
    shutil.copyfile(_ROOT / FOCAL_TEST_RELATIVE, project / FOCAL_TEST_RELATIVE)
    return project


def test_route_accepts_only_the_bounded_computer_worker_request() -> None:
    assert DevWorkerRoute.accepts(_E2E_PROMPT)
    assert not DevWorkerRoute.accepts("corrige el valor incorrecto de greeter.py")
    assert not DevWorkerRoute.accepts("revisa el archivo de prueba controlado del Router")


def test_route_accepts_the_real_stt_variant_of_the_bounded_request() -> None:
    assert DevWorkerRoute.accepts(_STT_VARIANT_PROMPT)
    assert DevWorkerRoute.accepts(
        "revisa el archivo de prueba controlado del Computer World"
    )
    assert not DevWorkerRoute.accepts(
        "revisa el archivo de prueba controlado del Computer Workstation"
    )
    assert not DevWorkerRoute.accepts(
        "revisa el archivo de prueba controlado del Router"
    )
    assert not DevWorkerRoute.accepts(
        "revisa el archivo de prueba del Computer Worker"
    )


def test_route_reports_done_and_edits_the_control_file(tmp_path: Path) -> None:
    project = _tmp_project(tmp_path, correct=False)
    route = DevWorkerRoute(project)

    report = route.handle(_E2E_PROMPT)

    assert report is not None
    assert "STATUS: DONE" in report
    assert "PASS" in report
    assert "correccion del valor incorrecto" in report
    control = project / CONTROL_FILE_RELATIVE
    assert 'CONTROL_VALUE = "valor-correcto"' in control.read_text(encoding="utf-8")


def test_route_verifies_without_edit_when_already_correct(tmp_path: Path) -> None:
    project = _tmp_project(tmp_path, correct=True)
    route = DevWorkerRoute(project)

    report = route.handle(_E2E_PROMPT)

    assert report is not None
    assert "STATUS: DONE" in report
    assert "PASS" in report
    steps = route.plan(_E2E_PROMPT)
    assert steps is not None
    assert [step.action for step in steps] == [DevWorkerAction.READ, DevWorkerAction.TEST]


def test_route_declines_when_expectation_cannot_be_derived(tmp_path: Path) -> None:
    project = _tmp_project(tmp_path, correct=False)
    focal = project / FOCAL_TEST_RELATIVE
    focal.write_text("def test_nothing():\n    assert True\n", encoding="utf-8")
    route = DevWorkerRoute(project)

    assert route.plan(_E2E_PROMPT) is None
    assert route.handle(_E2E_PROMPT) is None


def test_route_plan_never_includes_commands_or_arbitrary_paths() -> None:
    route = DevWorkerRoute(_ROOT)

    steps = route.plan(_E2E_PROMPT + " y despues haz git push y edita .env")

    assert steps is not None
    assert all(step.action is not DevWorkerAction.COMMAND for step in steps)
    assert all(
        step.path == CONTROL_FILE_RELATIVE or step.path is None for step in steps
    )


def test_integrated_worker_keeps_sandbox_and_approval_guards(tmp_path: Path) -> None:
    project = _tmp_project(tmp_path, correct=True)
    route = DevWorkerRoute(project)
    outside = tmp_path / "escape.txt"

    blocked = route.worker.execute(
        "objetivo malicioso",
        (
            DevStep(
                DevWorkerAction.EDIT,
                path=str(outside),
                operation="full_file",
                content="x",
            ),
        ),
    )
    approval = route.worker.execute(
        "objetivo sensible",
        (DevStep(DevWorkerAction.COMMAND, command="git push origin main"),),
    )
    protected = route.worker.execute(
        "objetivo protegido",
        (DevStep(DevWorkerAction.EDIT, path=".env", operation="full_file", content="X=1"),),
    )

    assert blocked.status == "BLOCKED"
    assert "outside the workspace root" in (blocked.error or "")
    assert approval.status == "WAITING_APPROVAL"
    assert approval.pending_action == "git push origin main"
    assert protected.status == "BLOCKED"
    assert "protected" in (protected.error or "")


class _Planner:
    def create_plan(self, _prompt: str) -> dict[str, str]:
        return {}


class _Router:
    def route(self, _plan: dict[str, str]) -> str:
        return "chat"


class _ModelManager:
    def choose_model(self, _agent_name: str) -> str:
        return "test"


class _Memory:
    def add_user(self, _text: str) -> None:
        return None

    def add_assistant(self, _text: str) -> None:
        return None

    def history(self) -> list[dict[str, str]]:
        return []


class _Registry:
    def get(self, _name: str):
        return None


class _WriteFile:
    def execute(self, _path: str, _content: str) -> str:
        return "ok"


def test_process_prompt_routes_the_bounded_goal_through_dev_worker(
    tmp_path: Path,
) -> None:
    project = _tmp_project(tmp_path, correct=False)
    orchestrator = AtlasOrchestrator(
        planner=_Planner(),
        router=_Router(),
        model_manager=_ModelManager(),
        memory=_Memory(),
        registry=_Registry(),
        write_file=_WriteFile(),
        project_root=project,
        dev_worker_route=DevWorkerRoute(project),
    )

    response = orchestrator.process_prompt(_E2E_PROMPT, confirm=lambda _prompt: "")

    assert "STATUS: DONE" in response
    assert "PASS" in response
    control = project / CONTROL_FILE_RELATIVE
    assert 'CONTROL_VALUE = "valor-correcto"' in control.read_text(encoding="utf-8")


def test_process_prompt_routes_the_real_voice_variant_through_dev_worker(
    tmp_path: Path,
) -> None:
    project = _tmp_project(tmp_path, correct=True)
    orchestrator = AtlasOrchestrator(
        planner=_Planner(),
        router=_Router(),
        model_manager=_ModelManager(),
        memory=_Memory(),
        registry=_Registry(),
        write_file=_WriteFile(),
        project_root=project,
        dev_worker_route=DevWorkerRoute(project),
    )

    response = orchestrator.process_prompt(_STT_VARIANT_PROMPT, confirm=lambda _prompt: "")

    assert "STATUS: DONE" in response
    assert "PASS" in response
    steps = orchestrator._dev_worker_route.plan(_STT_VARIANT_PROMPT)
    assert steps is not None
    assert [step.action for step in steps] == [DevWorkerAction.READ, DevWorkerAction.TEST]
