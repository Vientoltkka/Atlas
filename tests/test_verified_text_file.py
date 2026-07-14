from pathlib import Path
import os

import pytest

from bootstrap.bootstrap import Bootstrap
from tools.tool_context import ToolContext
from use_cases.action_engine import ActionEngineUseCase
from use_cases.desktop_interaction import DesktopInteractionUseCase
from use_cases.verified_text_file import CreateVerifiedTextFileUseCase
from use_cases.wait_engine import WaitEngine


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeDesktopExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []
        self.process_available_after = 1
        self.window_available_after = 1
        self.active_available_after = 1
        self.update_available_after = 1
        self.open_file_path: Path | None = None
        self.clipboard = ""
        self.selected = False
        self.fail_tool: str | None = None
        self.fail_after_save = False
        self.write_wrong_content = False
        self.shell_used = False
        self.content_executed = False

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ):
        self.calls.append((tool_name, context))

        if tool_name == self.fail_tool:
            raise RuntimeError(f"{tool_name} failed")

        if tool_name == "desktop.open_application":
            return "Visual Studio Code abierto."

        if tool_name == "desktop.list_processes":
            self.process_available_after -= 1
            if self.process_available_after <= 0:
                return [{"pid": 10, "name": "Code.exe"}]
            return []

        if tool_name == "desktop.list_windows":
            title = str(context.parameters["title"])
            self.window_available_after -= 1
            if self.window_available_after > 0:
                return []
            return [{"handle": 10, "title": f"{title} - Visual Studio Code"}]

        if tool_name == "desktop.open_folder":
            return "workspace abierto"

        if tool_name == "desktop.open_file":
            self.open_file_path = Path(str(context.parameters["path"]))
            if not self.open_file_path.exists():
                self.open_file_path.write_text("", encoding="utf-8")
            return "archivo abierto"

        if tool_name == "desktop.copy_clipboard_text":
            self.clipboard = str(context.parameters["text"])
            return len(self.clipboard)

        if tool_name == "desktop.activate_window":
            return "ventana activada"

        if tool_name == "desktop.get_foreground_window":
            self.active_available_after -= 1
            title = (
                "Visual Studio Code"
                if self.active_available_after <= 0
                else "Otra ventana"
            )
            return {"handle": 10, "title": title, "rect": (0, 0, 100, 100)}

        if tool_name == "desktop.press_hotkey":
            keys = context.parameters["keys"]
            if keys == ["ctrl", "a"]:
                self.selected = True
                return "seleccionado"
            if keys == ["ctrl", "v"]:
                return "pegado"
            if keys == ["ctrl", "s"]:
                assert self.open_file_path is not None
                content = "contenido incorrecto" if self.write_wrong_content else self.clipboard
                self.open_file_path.write_text(content, encoding="utf-8")
                stat = self.open_file_path.stat()
                os.utime(
                    self.open_file_path,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
                )
                if self.fail_after_save:
                    raise RuntimeError("save failed after write")
                return "guardado"

        return "ok"


def build_use_case(
    executor: FakeDesktopExecutor,
    max_content_chars: int = 100_000,
    max_retries: int = 2,
) -> CreateVerifiedTextFileUseCase:
    clock = ManualClock()
    wait_engine = WaitEngine(executor, clock.monotonic, clock.sleep)
    return CreateVerifiedTextFileUseCase(
        executor,
        ActionEngineUseCase(),
        wait_engine,
        max_content_chars=max_content_chars,
        max_retries=max_retries,
        timeout=1,
        poll_interval=0.1,
    )


def run_confirmed(
    use_case: CreateVerifiedTextFileUseCase,
    root: Path,
    target: str,
    content: str,
):
    return use_case.execute(root, target, content, confirm=lambda _: "s")


def test_validates_workspace_root_existing(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "ok")

    assert result.completed is True
    assert result.target_file == str((tmp_path / "nota.txt").resolve())


def test_rejects_missing_workspace_root(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    result = run_confirmed(build_use_case(executor), tmp_path / "missing", "a.txt", "ok")

    assert result.completed is False
    assert "workspace_root no existe" in result.summary
    assert executor.calls == []


def test_rejects_workspace_root_file(tmp_path: Path) -> None:
    root_file = tmp_path / "workspace.txt"
    root_file.write_text("", encoding="utf-8")
    executor = FakeDesktopExecutor()
    result = run_confirmed(build_use_case(executor), root_file, "a.txt", "ok")

    assert result.completed is False
    assert executor.calls == []


def test_rejects_path_traversal_and_outside_target(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    use_case = build_use_case(executor)

    traversal = run_confirmed(use_case, tmp_path, "..\\outside.txt", "ok")
    outside = run_confirmed(use_case, tmp_path, tmp_path.parent / "outside.md", "ok")

    assert traversal.completed is False
    assert outside.completed is False
    assert executor.calls == []


def test_accepts_txt_and_md(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    use_case = build_use_case(executor)

    txt = run_confirmed(use_case, tmp_path, "a.txt", "txt")
    md = run_confirmed(use_case, tmp_path, "b.md", "md")

    assert txt.completed is True
    assert md.completed is True


@pytest.mark.parametrize("name", ["a.py", "a.bat", "a.ps1", "a.cmd", "a.exe", "a.reg"])
def test_rejects_executable_extensions(tmp_path: Path, name: str) -> None:
    executor = FakeDesktopExecutor()
    result = run_confirmed(build_use_case(executor), tmp_path, name, "ok")

    assert result.completed is False
    assert executor.calls == []


def test_rejects_empty_non_str_and_too_large_content(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    use_case = build_use_case(executor, max_content_chars=3)

    empty = run_confirmed(use_case, tmp_path, "empty.txt", "")
    non_str = use_case.execute(tmp_path, "bad.txt", 123, confirm=lambda _: "s")
    too_large = run_confirmed(use_case, tmp_path, "large.txt", "abcd")

    assert empty.completed is False
    assert non_str.completed is False
    assert too_large.completed is False
    assert executor.calls == []


def test_confirmation_is_required_and_cancel_does_not_execute(tmp_path: Path) -> None:
    for answer in ("", "n"):
        executor = FakeDesktopExecutor()
        result = build_use_case(executor).execute(
            tmp_path,
            "nota.txt",
            "ok",
            confirm=lambda _: answer,
        )

        assert result.confirmed is False
        assert result.completed is False
        assert executor.calls == []
        assert not (tmp_path / "nota.txt").exists()


def test_preserves_original_state_and_detects_new_file(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("original", encoding="utf-8")
    executor = FakeDesktopExecutor()
    use_case = build_use_case(executor)

    existing_result = run_confirmed(use_case, tmp_path, "existing.txt", "new")
    new_result = run_confirmed(use_case, tmp_path, "new.md", "new file")

    assert existing_result.file_preexisted is True
    assert existing_result.file_created is False
    assert new_result.file_preexisted is False
    assert new_result.file_created is True


def test_executes_steps_in_order_and_uses_desktop_tools(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "ok")
    calls = [call[0] for call in executor.calls]

    assert result.completed is True
    assert result.workflow_name == "create_verified_text_file"
    assert calls[:5] == [
        "desktop.open_application",
        "desktop.list_processes",
        "desktop.list_windows",
        "desktop.open_folder",
        "desktop.open_file",
    ]
    assert "desktop.copy_clipboard_text" in calls
    assert "desktop.bring_window_to_front" in calls
    assert ("desktop.type_text" not in calls)
    assert ["ctrl", "a"] in [call[1].parameters.get("keys") for call in executor.calls]
    assert ["ctrl", "v"] in [call[1].parameters.get("keys") for call in executor.calls]
    assert ["ctrl", "s"] in [call[1].parameters.get("keys") for call in executor.calls]


def test_waits_process_window_activation_and_file_update(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "ok")
    step_names = [step.step_name for step in result.step_results]

    assert "Esperar proceso Code.exe" in step_names
    assert "Esperar ventana Visual Studio Code" in step_names
    assert "Esperar Visual Studio Code activo" in step_names
    assert "Esperar archivo actualizado" in step_names


def test_verifies_exact_content_and_returns_completed(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    content = "Atlas Phase 3.8 verified automation"
    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", content)

    assert result.completed is True
    assert result.verification_passed is True
    assert (tmp_path / "nota.txt").read_text(encoding="utf-8") == content
    assert result.rolled_back is False


def test_stops_on_first_failure_and_skips_later_steps(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    executor.fail_tool = "desktop.open_file"
    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "ok")
    calls = [call[0] for call in executor.calls]

    assert result.completed is False
    assert result.stopped_early is True
    assert result.step_results[-1].step_name == "Abrir archivo nota.txt"
    assert "desktop.copy_clipboard_text" not in calls


def test_retries_transient_window_failure_then_succeeds(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    executor.window_available_after = 12

    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "ok")

    assert result.completed is True
    assert result.retries_used == 1


def test_respects_max_retries_and_does_not_retry_validation(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    executor.window_available_after = 99

    result = run_confirmed(
        build_use_case(executor, max_retries=1),
        tmp_path,
        "nota.txt",
        "ok",
    )
    validation = run_confirmed(
        build_use_case(FakeDesktopExecutor(), max_retries=1),
        tmp_path / "missing",
        "nota.txt",
        "ok",
    )

    assert result.completed is False
    assert result.retries_used == 1
    assert validation.retries_used == 0


def test_restores_preexisting_file_after_failure(tmp_path: Path) -> None:
    file = tmp_path / "nota.txt"
    original = "original\nexacto"
    file.write_text(original, encoding="utf-8")
    executor = FakeDesktopExecutor()
    executor.fail_after_save = True

    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "nuevo")

    assert result.completed is False
    assert result.rolled_back is True
    assert result.rollback_failed is False
    assert file.read_text(encoding="utf-8") == original


def test_deletes_new_file_after_failure(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    executor.fail_after_save = True

    result = run_confirmed(build_use_case(executor), tmp_path, "nuevo.txt", "nuevo")

    assert result.completed is False
    assert result.rolled_back is True
    assert result.rollback_failed is False
    assert not (tmp_path / "nuevo.txt").exists()


def test_reports_rollback_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file = tmp_path / "nota.txt"
    file.write_text("original", encoding="utf-8")
    executor = FakeDesktopExecutor()
    executor.fail_after_save = True

    def fail_write_bytes(_: bytes) -> int:
        raise OSError("locked")

    monkeypatch.setattr(Path, "write_bytes", fail_write_bytes)

    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", "nuevo")

    assert result.rolled_back is True
    assert result.rollback_failed is True


def test_rollback_after_timeout_save_failure_and_wrong_verification(tmp_path: Path) -> None:
    timeout_executor = FakeDesktopExecutor()
    timeout_executor.process_available_after = 99
    timeout_result = run_confirmed(
        build_use_case(timeout_executor, max_retries=0),
        tmp_path,
        "timeout.txt",
        "nuevo",
    )

    save_file = tmp_path / "save.txt"
    save_file.write_text("original", encoding="utf-8")
    save_executor = FakeDesktopExecutor()
    save_executor.fail_after_save = True
    save_result = run_confirmed(build_use_case(save_executor), tmp_path, "save.txt", "nuevo")

    wrong_file = tmp_path / "wrong.txt"
    wrong_file.write_text("original", encoding="utf-8")
    wrong_executor = FakeDesktopExecutor()
    wrong_executor.write_wrong_content = True
    wrong_result = run_confirmed(build_use_case(wrong_executor), tmp_path, "wrong.txt", "nuevo")

    assert timeout_result.rolled_back is False
    assert save_result.rolled_back is True
    assert save_file.read_text(encoding="utf-8") == "original"
    assert wrong_result.rolled_back is True
    assert wrong_file.read_text(encoding="utf-8") == "original"


def test_does_not_modify_other_files_or_execute_content(tmp_path: Path) -> None:
    other = tmp_path / "other.txt"
    other.write_text("untouched", encoding="utf-8")
    executor = FakeDesktopExecutor()
    content = "print('no ejecutar')"

    result = run_confirmed(build_use_case(executor), tmp_path, "nota.txt", content)

    assert result.completed is True
    assert other.read_text(encoding="utf-8") == "untouched"
    assert executor.content_executed is False
    assert executor.shell_used is False


def test_interactive_command_collects_path_content_and_confirmation(tmp_path: Path) -> None:
    executor = FakeDesktopExecutor()
    use_case = build_use_case(executor)
    interaction = DesktopInteractionUseCase(
        executor,
        project_root=tmp_path,
        create_verified_text_file=use_case,
    )
    answers = iter(["nota.txt", "contenido", "s"])

    response = interaction.execute(
        "crea archivo verificado",
        confirm=lambda _: next(answers),
    )

    assert "create_verified_text_file" in str(response)
    assert "Completada: True" in str(response)
    assert (tmp_path / "nota.txt").read_text(encoding="utf-8") == "contenido"


def test_bootstrap_injects_verified_file_workflow() -> None:
    orchestrator = Bootstrap.build()

    assert (
        orchestrator
        ._desktop_interaction
        ._create_verified_text_file
        is not None
    )


def test_controlled_integration_success(tmp_path: Path) -> None:
    file = tmp_path / "nota.txt"
    file.write_text("inicial", encoding="utf-8")
    executor = FakeDesktopExecutor()

    result = run_confirmed(
        build_use_case(executor),
        tmp_path,
        "nota.txt",
        "contenido final",
    )

    assert result.completed is True
    assert result.verification_passed is True
    assert result.rolled_back is False
    assert file.read_text(encoding="utf-8") == "contenido final"


def test_controlled_integration_rollback(tmp_path: Path) -> None:
    file = tmp_path / "nota.txt"
    original = "contenido original"
    file.write_text(original, encoding="utf-8")
    executor = FakeDesktopExecutor()
    executor.fail_after_save = True

    result = run_confirmed(
        build_use_case(executor),
        tmp_path,
        "nota.txt",
        "contenido nuevo",
    )

    assert result.completed is False
    assert result.rolled_back is True
    assert file.read_text(encoding="utf-8") == original
