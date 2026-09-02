from pathlib import Path
from types import SimpleNamespace

from core.orchestrator import AtlasOrchestrator
from core.router import Router
from memory.conversation import ConversationMemory
from tools.tool_context import ToolContext
from use_cases.desktop_interaction import DesktopInteractionUseCase


class FakeToolExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ToolContext]] = []
        self.fail_on_execute = False
        self.clipboard_text: str | None = "Hola Atlas"
        self.processes: list[dict[str, object]] = []
        self.windows: list[dict[str, object]] = [
            {
                "handle": 10,
                "title": "Visual Studio Code - Atlas",
                "rect": (20, 30, 820, 630),
            }
        ]

    def execute(
        self,
        tool_name: str,
        context: ToolContext,
    ):
        if self.fail_on_execute:
            raise RuntimeError("tool failed")

        self.calls.append((tool_name, context))

        if tool_name == "desktop.get_screen_size":
            return (1920, 1080)

        if tool_name == "desktop.get_cursor_position":
            return (812, 430)

        if tool_name == "desktop.capture_screenshot":
            return str(Path(context.parameters["output_dir"]) / "shot.png")

        if tool_name == "desktop.copy_clipboard_text":
            text = context.parameters["text"]
            assert isinstance(text, str)
            self.clipboard_text = text
            return len(text)

        if tool_name == "desktop.read_clipboard_text":
            return self.clipboard_text

        if tool_name == "desktop.clear_clipboard":
            self.clipboard_text = None
            return "Portapapeles vaciado."

        if tool_name == "desktop.clipboard_has_text":
            return self.clipboard_text is not None

        if tool_name == "desktop.get_foreground_window":
            if not self.windows:
                raise RuntimeError("missing")

            return self.windows[0]

        if tool_name == "desktop.list_processes":
            query = str(context.parameters["query"]).lower()
            query = {
                "visual studio code": "code",
                "vs code": "code",
                "vscode": "code",
            }.get(query, query)
            return [
                process
                for process in self.processes
                if query in str(process["name"]).lower()
                or query in str(process["name"]).lower().replace(".exe", "")
            ]

        if tool_name == "desktop.get_process":
            pid = context.parameters["pid"]

            for process in self.processes:
                if process["pid"] == pid:
                    return process

            return None

        if tool_name == "desktop.close_application":
            return "Solicitud de cierre enviada a 1 ventana(s)."

        if tool_name == "desktop.terminate_process":
            return f"Proceso terminado: example.exe - PID {context.parameters['pid']}"

        if tool_name == "desktop.list_windows":
            title = str(context.parameters.get("title", "")).lower()
            return [
                window
                for window in self.windows
                if title in str(window["title"]).lower()
            ]

        if tool_name == "desktop.get_window_rect":
            handle = context.parameters["handle"]

            for window in self.windows:
                if window["handle"] == handle:
                    return window["rect"]

            raise RuntimeError("missing")

        return "ok"


def test_desktop_interaction_opens_application() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Abre Visual Studio Code")

    assert result == "\u2713 Abriendo Visual Studio Code."
    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {
        "application": "Visual Studio Code"
    }


def test_desktop_interaction_opens_application_with_english_alias() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("open Chrome")

    assert result == "\u2713 Abriendo Chrome."
    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {"application": "Chrome"}


def test_desktop_interaction_opens_vscode_from_conversational_name() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    use_case.execute("abre VS Code")

    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {"application": "VS Code"}


def test_desktop_interaction_resolves_bloc_de_notas_alias() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("Abre notepad") == "✓ Abriendo notepad."
    result = use_case.execute("Abre el Bloc de notas")

    assert result == "✓ Abriendo notepad."
    assert [call[0] for call in executor.calls] == [
        "desktop.open_application",
        "desktop.open_application",
    ]
    assert [call[1].parameters for call in executor.calls] == [
        {"application": "notepad"},
        {"application": "notepad"},
    ]


def test_desktop_interaction_routes_powershell_and_explorer_to_open_application() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    powershell = use_case.execute("Abre PowerShell")
    explorer = use_case.execute("Abre el explorador de archivos")

    assert powershell == "\u2713 Abriendo PowerShell."
    assert explorer == "\u2713 Abriendo el explorador de archivos."
    assert [call[0] for call in executor.calls] == [
        "desktop.open_application",
        "desktop.open_application",
    ]
    assert [call[1].parameters for call in executor.calls] == [
        {"application": "PowerShell"},
        {"application": "el explorador de archivos"},
    ]


class UnknownApplicationExecutor(FakeToolExecutor):
    def execute(self, tool_name: str, context: ToolContext):
        self.calls.append((tool_name, context))
        raise FileNotFoundError("missing")


def test_desktop_interaction_reports_unknown_application_specifically() -> None:
    executor = UnknownApplicationExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Abre Aplicación inexistente")

    assert result == "No encuentro esa aplicación."
    assert executor.calls[0][0] == "desktop.open_application"


class FailedApplicationExecutor(FakeToolExecutor):
    def execute(self, tool_name: str, context: ToolContext):
        self.calls.append((tool_name, context))
        raise OSError("launch rejected")


def test_desktop_interaction_reports_application_launch_failure() -> None:
    executor = FailedApplicationExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Abre Visual Studio Code")

    assert result == "No pude abrir Visual Studio Code."
    assert executor.calls[0][0] == "desktop.open_application"
def test_desktop_interaction_does_not_treat_read_or_destructive_requests_as_open() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute(r"Lee C:\AI\Atlas\README.md") is None
    assert use_case.execute(r"Borra C:\AI\Atlas\README.md") is None
    assert executor.calls == []


def test_desktop_interaction_rejects_incomplete_open_command() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("abre")

    assert result == "Error: Falta el objetivo a abrir."
    assert executor.calls == []


def test_desktop_interaction_rejects_arbitrary_command_execution() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("ejecuta dir")

    assert result == "Error: No se aceptan comandos arbitrarios."
    assert executor.calls == []


def test_desktop_interaction_rejects_arbitrary_command_with_atlas_prefix() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Atlas, ejecuta dir")

    assert result == "Error: No se aceptan comandos arbitrarios."
    assert executor.calls == []


def test_desktop_interaction_checks_process_running() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 4820,
            "name": "Code.exe",
            "executable_path": None,
            "window_titles": ("Atlas - Visual Studio Code",),
            "is_running": True,
        },
        {
            "pid": 9404,
            "name": "Code.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("¿está abierto Code?")

    assert result == "\u2713 Code esta abierto.\nProcesos: 2\nPID: 4820, 9404"
    assert executor.calls[0][0] == "desktop.list_processes"


def test_desktop_interaction_reports_process_absent() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("is Chrome running")

    assert result == "Chrome no esta abierto."


def test_desktop_interaction_lists_processes() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": ("Chrome",),
            "is_running": True,
        }
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("lista los procesos de chrome")

    assert result == (
        "Procesos encontrados:\n\n"
        "1. chrome.exe - PID 20 - ventanas visibles: si"
    )


def test_desktop_interaction_shows_process_pids() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        }
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("muestra el PID de chrome")

    assert result == "PID: 20"


def test_desktop_interaction_close_application_requires_confirmation() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": ("Chrome",),
            "is_running": True,
        }
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Chrome")

    assert result == "Cierre cancelado."
    assert [call[0] for call in executor.calls] == ["desktop.list_processes"]


def test_desktop_interaction_close_application_cancels_with_empty_or_no() -> None:
    for response in ("", "n"):
        executor = FakeToolExecutor()
        executor.processes = [
            {
                "pid": 20,
                "name": "chrome.exe",
                "executable_path": None,
                "window_titles": ("Chrome",),
                "is_running": True,
            }
        ]
        use_case = DesktopInteractionUseCase(executor)

        result = use_case.execute("close Chrome", confirm=lambda _: response)

        assert result == "Cierre cancelado."
        assert all(call[0] != "desktop.close_application" for call in executor.calls)


def test_desktop_interaction_close_application_after_confirmation() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": ("Chrome",),
            "is_running": True,
        }
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Chrome", confirm=lambda _: "s")

    assert result == "\u2713 Solicitud de cierre enviada: chrome.exe - PID 20"
    assert executor.calls[1][0] == "desktop.close_application"
    assert executor.calls[1][1].parameters == {"pid": 20}


def test_desktop_interaction_close_application_requires_explicit_selection() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
        {
            "pid": 21,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Chrome")

    assert result == "Error: Varias coincidencias. Se requiere seleccion explicita."
    assert all(call[0] != "desktop.close_application" for call in executor.calls)


def test_desktop_interaction_close_application_selects_process() -> None:
    answers = iter(["2", "yes"])
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
        {
            "pid": 21,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Chrome", confirm=lambda _: next(answers))

    assert result == "\u2713 Solicitud de cierre enviada: chrome.exe - PID 21"
    assert executor.calls[-1][1].parameters == {"pid": 21}


def test_desktop_interaction_close_application_rejects_invalid_selection() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 20,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
        {
            "pid": 21,
            "name": "chrome.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        },
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Chrome", confirm=lambda _: "3")

    assert result == "Error: Seleccion invalida."
    assert all(call[0] != "desktop.close_application" for call in executor.calls)


def test_desktop_interaction_terminate_process_validates_pid() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("termina PID abc") == "Error: PID invalido."
    assert use_case.execute("termina el proceso con PID 0") == "Error: PID invalido."
    assert use_case.execute("termina el proceso con PID 4") == "Error: Proceso protegido."
    assert executor.calls == []


def test_desktop_interaction_terminate_process_requires_confirmation() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 1234,
            "name": "example.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        }
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("terminate process 1234")

    assert result == "Terminacion cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_process"]


def test_desktop_interaction_terminate_process_after_confirmation() -> None:
    executor = FakeToolExecutor()
    executor.processes = [
        {
            "pid": 1234,
            "name": "example.exe",
            "executable_path": None,
            "window_titles": (),
            "is_running": True,
        }
    ]
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mata el proceso 1234", confirm=lambda _: "s")

    assert result == "\u2713 Proceso terminado: example.exe - PID 1234"
    assert [call[0] for call in executor.calls] == [
        "desktop.get_process",
        "desktop.terminate_process",
    ]


def test_desktop_interaction_opens_existing_folder(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    folder = tmp_path / "core"
    folder.mkdir()

    result = use_case.execute("Abre core")

    assert result == f"\u2713 Abriendo {folder}."
    assert executor.calls[0][0] == "desktop.open_folder"
    assert executor.calls[0][1].parameters == {"path": str(folder)}


def test_desktop_interaction_opens_folder_with_natural_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    folder = tmp_path / "Atlas"
    folder.mkdir()

    result = use_case.execute(f"Abre la carpeta {folder}")

    assert result == f"\u2713 Abriendo {folder}."
    assert executor.calls[0][0] == "desktop.open_folder"


def test_desktop_interaction_opens_folder_with_atlas_invocation_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    folder = tmp_path / "Atlas"
    folder.mkdir()

    result = use_case.execute(f"Atlas, abre {folder}")

    assert result == f"\u2713 Abriendo {folder}."
    assert executor.calls[0][0] == "desktop.open_folder"
    assert executor.calls[0][1].parameters == {"path": str(folder)}


def test_desktop_interaction_opens_file_with_atlas_invocation_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "core" / "router.py"
    file.parent.mkdir()
    file.write_text("print('demo')", encoding="utf-8")

    result = use_case.execute(f"Atlas abre {file}")

    assert result == f"\u2713 Abriendo {file}."
    assert executor.calls[0][0] == "desktop.open_file"
    assert executor.calls[0][1].parameters == {"path": str(file)}


def test_desktop_interaction_reports_missing_path_with_atlas_invocation_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    result = use_case.execute("Atlas, abre missing.py")

    assert result == "La ruta no existe."
    assert executor.calls == []


def test_desktop_interaction_preserves_folder_path_with_spaces(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    folder = tmp_path / "Atlas Workspace"
    folder.mkdir()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    use_case.execute(f"abre la carpeta {folder}")

    assert executor.calls[0][0] == "desktop.open_folder"
    assert executor.calls[0][1].parameters == {"path": str(folder)}


def test_desktop_interaction_opens_existing_file(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "core" / "router.py"
    file.parent.mkdir()
    file.write_text("print('demo')", encoding="utf-8")

    result = use_case.execute("Abre core/router.py")

    assert result == f"\u2713 Abriendo {file}."
    assert executor.calls[0][0] == "desktop.open_file"
    assert executor.calls[0][1].parameters == {"path": str(file)}


def test_desktop_interaction_distinguishes_file_path_from_folder(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    file = tmp_path / "main.py"
    file.write_text("print('Atlas')", encoding="utf-8")
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    use_case.execute(f"abre {file}")

    assert executor.calls[0][0] == "desktop.open_file"
    assert executor.calls[0][1].parameters == {"path": str(file)}


def test_desktop_interaction_opens_file_with_natural_prefix(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "core" / "router.py"
    file.parent.mkdir()
    file.write_text("print('demo')", encoding="utf-8")

    result = use_case.execute("Abre el archivo core/router.py")

    assert result == f"\u2713 Abriendo {file}."
    assert executor.calls[0][0] == "desktop.open_file"


def test_desktop_interaction_reports_missing_file(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    result = use_case.execute("Abre missing.py")

    assert result == "La ruta no existe."
    assert executor.calls == []


def test_desktop_interaction_reports_path_kind_mismatches(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    folder = tmp_path / "folder"
    folder.mkdir()
    file = tmp_path / "file.txt"
    file.write_text("Atlas", encoding="utf-8")

    assert use_case.execute(f"Abre la carpeta {file}") == (
        "La ruta es un archivo, no una carpeta."
    )
    assert use_case.execute(f"Abre el archivo {folder}") == (
        "La ruta es una carpeta, no un archivo."
    )
    assert executor.calls == []


def test_desktop_interaction_types_text() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute('Escribe: print("Hola")')

    assert result == "\u2713 Texto escrito."
    assert executor.calls[0][0] == "desktop.type_text"
    assert executor.calls[0][1].parameters == {"text": 'print("Hola")'}


def test_desktop_interaction_copies_clipboard_text() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("copia al portapapeles: Hola Atlas")

    assert result == "\u2713 Texto copiado al portapapeles.\nCaracteres: 10"
    assert executor.clipboard_text == "Hola Atlas"
    assert executor.calls[0][0] == "desktop.copy_clipboard_text"
    assert executor.calls[0][1].parameters == {"text": "Hola Atlas"}


def test_desktop_interaction_preserves_clipboard_copy_text_exactly() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    text = "LÃ­nea 1\n  LÃ­nea 2: Ã±"
    result = use_case.execute(f"copia este texto al portapapeles: {text}")

    assert result.endswith(f"Caracteres: {len(text)}")
    assert executor.calls[0][1].parameters == {"text": text}


def test_desktop_interaction_copies_text_before_clipboard_suffix() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    use_case.execute("copia hola mundo al portapapeles")

    assert executor.calls[0][0] == "desktop.copy_clipboard_text"
    assert executor.calls[0][1].parameters == {"text": "hola mundo"}


def test_desktop_interaction_routes_minimum_active_window_text_commands() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("Escribe el texto Prueba operativa") == "✓ Texto escrito."
    assert executor.calls[-1][0] == "desktop.type_text"
    assert executor.calls[-1][1].parameters == {"text": "Prueba operativa"}

    assert use_case.execute("Copia este texto: Atlas clipboard OK").endswith("Caracteres: 18")
    assert executor.calls[-1][0] == "desktop.copy_clipboard_text"
    assert executor.calls[-1][1].parameters == {"text": "Atlas clipboard OK"}

    assert use_case.execute("Pega el texto", confirm=lambda _: "s") == "✓ Contenido pegado."
    assert executor.calls[-1][0] == "desktop.paste_clipboard"
    assert executor.calls[-1][1].parameters == {}


def test_desktop_interaction_copies_clipboard_text_without_colon() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("copiar al portapapeles Hola Atlas")

    assert result.endswith("Caracteres: 10")
    assert executor.clipboard_text == "Hola Atlas"


def test_desktop_interaction_rejects_incomplete_clipboard_copy() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("copia al portapapeles")

    assert result == "Error: Orden incompleta: falta el texto a copiar."
    assert executor.calls == []


def test_desktop_interaction_rejects_unsupported_clipboard_format() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("lee imagen del portapapeles")

    assert result == "Error: Formato de portapapeles no soportado."
    assert executor.calls == []


def test_desktop_interaction_reads_clipboard_text() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("que hay en el portapapeles")

    assert result == "Contenido del portapapeles:\n\nHola Atlas"
    assert executor.calls[0][0] == "desktop.read_clipboard_text"


def test_desktop_interaction_reads_clipboard_from_conversational_question() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    use_case.execute("qué tengo en el portapapeles")

    assert executor.calls[0][0] == "desktop.read_clipboard_text"


def test_desktop_interaction_reports_clipboard_without_text() -> None:
    executor = FakeToolExecutor()
    executor.clipboard_text = None
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("read clipboard")

    assert result == "El portapapeles no contiene texto."


def test_desktop_interaction_checks_whether_clipboard_has_text() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clipboard has text")

    assert result == "El portapapeles contiene texto."
    assert executor.calls[0][0] == "desktop.clipboard_has_text"


def test_desktop_interaction_reports_clipboard_has_no_text() -> None:
    executor = FakeToolExecutor()
    executor.clipboard_text = None
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("hay texto en el portapapeles")

    assert result == "El portapapeles no contiene texto."


def test_desktop_interaction_clear_clipboard_requires_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("limpia el portapapeles")

    assert result == "Limpieza cancelada."
    assert executor.clipboard_text == "Hola Atlas"
    assert executor.calls == []


def test_desktop_interaction_clear_clipboard_cancels_with_empty_or_no() -> None:
    for response in ("", "n"):
        executor = FakeToolExecutor()
        use_case = DesktopInteractionUseCase(executor)

        result = use_case.execute(
            "vacia el portapapeles",
            confirm=lambda _: response,
        )

        assert result == "Limpieza cancelada."
        assert executor.clipboard_text == "Hola Atlas"
        assert executor.calls == []


def test_desktop_interaction_clear_clipboard_after_yes() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clear clipboard", confirm=lambda _: "s")

    assert result == "\u2713 Portapapeles vaciado."
    assert executor.clipboard_text is None
    assert executor.calls[0][0] == "desktop.clear_clipboard"


def test_desktop_interaction_paste_requires_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("pega el portapapeles")

    assert result == "Pegado cancelado."
    assert [call[0] for call in executor.calls] == ["desktop.clipboard_has_text"]


def test_desktop_interaction_paste_cancels_with_empty_or_no() -> None:
    for response in ("", "n"):
        executor = FakeToolExecutor()
        use_case = DesktopInteractionUseCase(executor)

        result = use_case.execute(
            "paste clipboard",
            confirm=lambda _: response,
        )

        assert result == "Pegado cancelado."
        assert all(call[0] != "desktop.paste_clipboard" for call in executor.calls)


def test_desktop_interaction_does_not_paste_without_text() -> None:
    executor = FakeToolExecutor()
    executor.clipboard_text = None
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("pega el contenido del portapapeles", confirm=lambda _: "s")

    assert result == "Error: El portapapeles no contiene texto."
    assert [call[0] for call in executor.calls] == ["desktop.clipboard_has_text"]


def test_desktop_interaction_pastes_without_window_selection() -> None:
    executor = FakeToolExecutor()
    executor.windows = []
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("pega", confirm=lambda _: "s")

    assert result == "✓ Contenido pegado."
    assert [call[0] for call in executor.calls] == ["desktop.clipboard_has_text", "desktop.paste_clipboard"]


def test_desktop_interaction_pastes_with_ctrl_v_tool_after_yes() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("pega el portapapeles", confirm=lambda _: "yes")

    assert result == "\u2713 Contenido pegado."
    assert [call[0] for call in executor.calls] == [
        "desktop.clipboard_has_text",
        "desktop.paste_clipboard",
    ]
    assert executor.calls[1][1].parameters == {}


def test_desktop_interaction_saves_file() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Guarda el archivo")

    assert result == "\u2713 Archivo guardado."
    assert executor.calls[0][0] == "desktop.save_file"
    assert executor.calls[0][1].parameters == {
        "window_title": "Visual Studio Code"
    }


def test_desktop_interaction_ignores_unknown_command() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("Analiza router.py")

    assert result is None
    assert executor.calls == []


def test_desktop_interaction_does_not_handle_normal_conversation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    assert use_case.execute("qué es una sentadilla frontal") is None
    assert executor.calls == []


def test_desktop_interaction_gets_screen_size() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("tamaño de pantalla")

    assert result == "\u2713 Tamaño de pantalla: 1920 x 1080."
    assert executor.calls[0][0] == "desktop.get_screen_size"


def test_desktop_interaction_gets_cursor_position() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("posición del ratón")

    assert result == "\u2713 Posición actual: (812, 430)."
    assert executor.calls[0][0] == "desktop.get_cursor_position"


def test_desktop_interaction_moves_cursor_with_comma_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve el ratón a 500, 300")

    assert result == "\u2713 Cursor movido a (500, 300)."
    assert executor.calls[0][0] == "desktop.move_cursor"
    assert executor.calls[0][1].parameters == {"x": 500, "y": 300}


def test_desktop_interaction_moves_cursor_with_space_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve el cursor a 500 300")

    assert result == "\u2713 Cursor movido a (500, 300)."
    assert executor.calls[0][0] == "desktop.move_cursor"


def test_desktop_interaction_rejects_incomplete_move_command() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve el ratón")

    assert result == "Error: Orden incompleta: faltan coordenadas."
    assert executor.calls == []


def test_desktop_interaction_rejects_non_numeric_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en abc, 300", confirm=lambda _: "s")

    assert result == "Error: Las coordenadas deben ser numericas."
    assert executor.calls == []


def test_desktop_interaction_rejects_negative_click_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en -1, 300", confirm=lambda _: "s")

    assert result == "Error: Las coordenadas no pueden ser negativas."
    assert executor.calls == []


def test_desktop_interaction_rejects_out_of_screen_click_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en 99999, 99999", confirm=lambda _: "s")

    assert result == "Error: Coordenadas fuera de pantalla: (99999, 99999)."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_left_click_after_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic en 500, 300", confirm=lambda _: "s")

    assert result == "\u2713 Clic realizado en (500, 300)."
    assert executor.calls[0][0] == "desktop.get_screen_size"
    assert executor.calls[1][0] == "desktop.left_click"
    assert executor.calls[1][1].parameters == {"x": 500, "y": 300}


def test_desktop_interaction_double_click_after_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "haz doble clic en 500, 300",
        confirm=lambda _: "si",
    )

    assert result == "\u2713 Doble clic realizado en (500, 300)."
    assert executor.calls[0][0] == "desktop.get_screen_size"
    assert executor.calls[1][0] == "desktop.double_click"


def test_desktop_interaction_right_click_after_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "haz clic derecho en 500, 300",
        confirm=lambda _: "yes",
    )

    assert result == "\u2713 Clic derecho realizado en (500, 300)."
    assert executor.calls[0][0] == "desktop.get_screen_size"
    assert executor.calls[1][0] == "desktop.right_click"


def test_desktop_interaction_cancels_click_with_empty_response() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clic en 500 300", confirm=lambda _: "")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_cancels_click_with_no_response() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clic en 500 300", confirm=lambda _: "n")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_click_requires_confirmation_callback() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("clic en 500 300")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == ["desktop.get_screen_size"]


def test_desktop_interaction_scrolls_down() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("desplázate hacia abajo")

    assert result == "\u2713 Scroll hacia abajo."
    assert executor.calls[0][0] == "desktop.scroll_vertical"
    assert executor.calls[0][1].parameters == {"direction": "down"}


def test_desktop_interaction_scrolls_up() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("scroll arriba")

    assert result == "\u2713 Scroll hacia arriba."
    assert executor.calls[0][0] == "desktop.scroll_vertical"
    assert executor.calls[0][1].parameters == {"direction": "up"}


def test_desktop_interaction_rejects_incomplete_click() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("haz clic")

    assert result == "Error: Orden incompleta: faltan coordenadas."
    assert executor.calls == []


def test_desktop_interaction_captures_screenshot(
    tmp_path: Path,
) -> None:
    executor = FakeToolExecutor()
    screenshots = tmp_path / "screenshots"
    use_case = DesktopInteractionUseCase(
        executor,
        screenshots_dir=screenshots,
    )

    result = use_case.execute("haz una captura de pantalla")

    assert result == f"\u2713 Captura guardada en:\n{screenshots / 'shot.png'}"
    assert executor.calls[0][0] == "desktop.capture_screenshot"
    assert executor.calls[0][1].parameters == {
        "output_dir": str(screenshots)
    }


def test_desktop_interaction_rejects_region_capture() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("captura una región")

    assert result == "Error: Solo se admite captura de pantalla completa."
    assert executor.calls == []


def test_desktop_interaction_reports_screenshot_failure() -> None:
    executor = FakeToolExecutor()
    executor.fail_on_execute = True
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("screenshot")

    assert result == "Error: tool failed"


def test_desktop_interaction_lists_windows() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("lista ventanas de Visual Studio Code")

    assert result == "Se encontraron ventanas:\n1. Visual Studio Code - Atlas"
    assert executor.calls[0][0] == "desktop.list_windows"


def test_desktop_interaction_reports_missing_window() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("maximiza Notepad")

    assert result == "Error: No se encontraron ventanas para 'Notepad'."
    assert executor.calls[0][0] == "desktop.list_windows"


def test_desktop_interaction_detects_multiple_windows_without_selection() -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {
            "handle": 11,
            "title": "Visual Studio Code - Tests",
            "rect": (40, 50, 840, 650),
        }
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("maximiza Visual Studio Code")

    assert result == (
        "Error: Varias ventanas coinciden. "
        "Se requiere seleccion explicita."
    )
    assert [call[0] for call in executor.calls] == ["desktop.list_windows"]


def test_desktop_interaction_selects_ambiguous_window() -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {
            "handle": 11,
            "title": "Visual Studio Code - Tests",
            "rect": (40, 50, 840, 650),
        }
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("maximiza Visual Studio Code", confirm=lambda _: "2")

    assert result == "\u2713 Ventana maximizada:\nVisual Studio Code - Tests"
    assert executor.calls[0][0] == "desktop.list_windows"
    assert executor.calls[1][0] == "desktop.maximize_window"
    assert executor.calls[1][1].parameters == {"handle": 11}


def test_desktop_interaction_rejects_invalid_window_selection() -> None:
    executor = FakeToolExecutor()
    executor.windows.append(
        {
            "handle": 11,
            "title": "Visual Studio Code - Tests",
            "rect": (40, 50, 840, 650),
        }
    )
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("maximiza Visual Studio Code", confirm=lambda _: "3")

    assert result == "Error: Seleccion invalida."
    assert [call[0] for call in executor.calls] == ["desktop.list_windows"]


def test_desktop_interaction_maximizes_window() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("maximiza Visual Studio Code")

    assert result == "\u2713 Ventana maximizada:\nVisual Studio Code - Atlas"
    assert executor.calls[1][0] == "desktop.maximize_window"


def test_desktop_interaction_minimizes_window() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("minimiza Visual Studio Code")

    assert result == "\u2713 Ventana minimizada:\nVisual Studio Code - Atlas"
    assert executor.calls[1][0] == "desktop.minimize_window"


def test_desktop_interaction_restores_window() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("restaura Visual Studio Code")

    assert result == "\u2713 Ventana restaurada:\nVisual Studio Code - Atlas"
    assert executor.calls[1][0] == "desktop.restore_window"


def test_desktop_interaction_brings_window_to_front_with_aliases() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("trae Visual Studio Code al frente")

    assert result == "\u2713 Ventana activada:\nVisual Studio Code - Atlas"
    assert executor.calls[1][0] == "desktop.bring_window_to_front"


def test_desktop_interaction_activates_window_alias() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("activa Visual Studio Code")

    assert result == "\u2713 Ventana activada:\nVisual Studio Code - Atlas"
    assert executor.calls[1][0] == "desktop.bring_window_to_front"


def test_desktop_interaction_moves_window_with_comma_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve Visual Studio Code a 100, 100")

    assert result == "\u2713 Ventana movida a (100, 100)."
    assert executor.calls[1][0] == "desktop.move_window"
    assert executor.calls[1][1].parameters == {
        "handle": 10,
        "x": 100,
        "y": 100,
    }


def test_desktop_interaction_moves_window_with_space_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve Visual Studio Code a 100 100")

    assert result == "\u2713 Ventana movida a (100, 100)."
    assert executor.calls[1][0] == "desktop.move_window"


def test_desktop_interaction_resizes_window() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "cambia el tamaño de Visual Studio Code a 1200, 800"
    )

    assert result == "\u2713 Ventana redimensionada a 1200 x 800."
    assert executor.calls[1][0] == "desktop.resize_window"


def test_desktop_interaction_resizes_window_with_alias() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("redimensiona Visual Studio Code a 1200 800")

    assert result == "\u2713 Ventana redimensionada a 1200 x 800."
    assert executor.calls[1][0] == "desktop.resize_window"


def test_desktop_interaction_moves_and_resizes_window() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "mueve y cambia el tamaño de Visual Studio Code a 100, 100, 1200, 800"
    )

    assert result == (
        "\u2713 Ventana movida a (100, 100) "
        "y redimensionada a 1200 x 800."
    )
    assert executor.calls[1][0] == "desktop.move_resize_window"


def test_desktop_interaction_rejects_incomplete_window_command() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("maximiza")

    assert result == "Error: Orden incompleta: falta el titulo de ventana."
    assert executor.calls == []


def test_desktop_interaction_rejects_invalid_window_coordinates() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("mueve Visual Studio Code a abc, 100")

    assert result == "Error: Los parametros deben ser numericos."
    assert executor.calls == []


def test_desktop_interaction_rejects_incomplete_move_resize() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute(
        "mueve y cambia el tamaño de Visual Studio Code a 100, 100, 1200"
    )

    assert result == "Error: Numero de parametros invalido."
    assert executor.calls == []


def test_desktop_interaction_close_requires_confirmation() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Visual Studio Code")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == [
        "desktop.list_processes",
        "desktop.list_windows",
    ]


def test_desktop_interaction_closes_after_yes() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Visual Studio Code", confirm=lambda _: "s")

    assert result == "\u2713 Solicitud de cierre enviada."
    assert executor.calls[2][0] == "desktop.close_window"


def test_desktop_interaction_closes_after_si() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Visual Studio Code", confirm=lambda _: "sí")

    assert result == "\u2713 Solicitud de cierre enviada."
    assert executor.calls[2][0] == "desktop.close_window"


def test_desktop_interaction_cancels_close_with_empty_response() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Visual Studio Code", confirm=lambda _: "")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == [
        "desktop.list_processes",
        "desktop.list_windows",
    ]


def test_desktop_interaction_cancels_close_with_no_response() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)

    result = use_case.execute("cierra Visual Studio Code", confirm=lambda _: "n")

    assert result == "Acción cancelada."
    assert [call[0] for call in executor.calls] == [
        "desktop.list_processes",
        "desktop.list_windows",
    ]

class _ExistingPath:
    def __init__(self, value: str, *, is_dir: bool) -> None:
        self._value = value
        self._is_dir = is_dir

    def exists(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return self._is_dir

    def is_file(self) -> bool:
        return not self._is_dir

    def __str__(self) -> str:
        return self._value


def test_desktop_e2e_opens_vs_code_alias() -> None:
    executor = FakeToolExecutor()
    result = DesktopInteractionUseCase(executor).execute("Abre VS Code")

    assert result == "✓ Abriendo VS Code."
    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {"application": "VS Code"}


def test_desktop_e2e_opens_explicit_existing_folder() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)
    use_case._resolve_path = lambda _target: _ExistingPath(r"C:\AI\Atlas", is_dir=True)  # type: ignore[method-assign]

    assert use_case.execute(r"Abre la carpeta C:\AI\Atlas") == r"✓ Abriendo C:\AI\Atlas."
    assert executor.calls[0][0] == "desktop.open_folder"


def test_desktop_e2e_opens_existing_path_as_folder() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)
    use_case._resolve_path = lambda _target: _ExistingPath(r"C:\AI\Atlas", is_dir=True)  # type: ignore[method-assign]

    assert use_case.execute(r"Abre C:\AI\Atlas") == r"✓ Abriendo C:\AI\Atlas."
    assert executor.calls[0][0] == "desktop.open_folder"


def test_desktop_e2e_keeps_existing_file_as_open_file() -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor)
    use_case._resolve_path = lambda _target: _ExistingPath(r"C:\AI\Atlas\README.md", is_dir=False)  # type: ignore[method-assign]

    assert use_case.execute(r"Abre C:\AI\Atlas\README.md") == r"✓ Abriendo C:\AI\Atlas\README.md."
    assert executor.calls[0][0] == "desktop.open_file"


def test_orchestrator_prioritizes_registered_desktop_folder() -> None:
    executor = FakeToolExecutor()
    desktop = DesktopInteractionUseCase(executor)

    class CapabilityGap:
        pending_confirmation_id = None

        def handle(self, _prompt: str):
            raise AssertionError("desktop.open_folder must not reach capability gap")

    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task="chat", objective=prompt)),
        router=Router(), model_manager=SimpleNamespace(), memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda _name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        desktop_interaction=desktop, execution_conversation=CapabilityGap(),
    )

    assert orchestrator.process_prompt(r"Abre la carpeta C:\AI\Atlas", confirm=lambda _: "") == r"✓ Abriendo C:\AI\Atlas."
    assert executor.calls[0][0] == "desktop.open_folder"


def test_orchestrator_opens_registered_desktop_folder_with_atlas_prefix() -> None:
    executor = FakeToolExecutor()
    desktop = DesktopInteractionUseCase(executor)

    class CapabilityGap:
        pending_confirmation_id = None

        def handle(self, _prompt: str):
            raise AssertionError("desktop.open_folder must not reach capability gap")

    orchestrator = AtlasOrchestrator(
        planner=SimpleNamespace(create_plan=lambda prompt: SimpleNamespace(task="chat", objective=prompt)),
        router=Router(), model_manager=SimpleNamespace(), memory=ConversationMemory(),
        registry=SimpleNamespace(get=lambda _name: None),
        write_file=SimpleNamespace(execute=lambda *_args: "unused"),
        desktop_interaction=desktop, execution_conversation=CapabilityGap(),
    )

    assert orchestrator.process_prompt(r"Atlas, abre C:\AI\Atlas", confirm=lambda _: "") == r"✓ Abriendo C:\AI\Atlas."
    assert executor.calls[0][0] == "desktop.open_folder"



def test_desktop_interaction_opens_file_with_application_alias(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "notas.txt"
    file.write_text("demo", encoding="utf-8")

    result = use_case.execute(f"abre {file} con el bloc de notas")

    assert result == f"\u2713 Abriendo {file} con notepad."
    assert executor.calls[0][0] == "desktop.open_file"
    assert executor.calls[0][1].parameters == {"path": str(file), "application": "notepad"}


def test_desktop_interaction_without_application_keeps_previous_behavior(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)
    file = tmp_path / "notas.txt"
    file.write_text("demo", encoding="utf-8")

    use_case.execute(f"abre {file}")

    assert executor.calls[0][0] == "desktop.open_file"
    assert executor.calls[0][1].parameters == {"path": str(file)}



def test_desktop_interaction_opens_known_application_by_name_with_article(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    use_case.execute("abre la calculadora")

    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {"application": "calculadora"}


def test_desktop_interaction_opens_vs_code_by_name_with_article(tmp_path: Path) -> None:
    executor = FakeToolExecutor()
    use_case = DesktopInteractionUseCase(executor, project_root=tmp_path)

    use_case.execute("abre el vs code")

    assert executor.calls[0][0] == "desktop.open_application"
    assert executor.calls[0][1].parameters == {"application": "vs code"}
