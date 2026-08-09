from __future__ import annotations

from tools.execution_coordinator import (
    ExecutionCoordinationResult,
    ExecutionCoordinationStatus,
)
from tools.execution_decision import ExecutionDecision, ExecutionMode
from tools.intent_selector import ToolIntent
from tools.single_tool_runner import ToolRunResult
from tools.tool_chain_runner import ToolChainResult, ToolChainStepResult
from use_cases.execution_result_presenter import (
    ExecutionResultPresenter,
    PresentationLimits,
)


def _decision(mode: ExecutionMode = ExecutionMode.SINGLE_TOOL) -> ExecutionDecision:
    return ExecutionDecision(mode=mode, reason="test", confidence=1.0)


def _coordination(
    status: ExecutionCoordinationStatus,
    execution_result=None,
    *,
    mode: ExecutionMode = ExecutionMode.SINGLE_TOOL,
    message: str = "test",
    validation_errors: tuple[str, ...] = (),
    confirmation_id: str | None = None,
) -> ExecutionCoordinationResult:
    return ExecutionCoordinationResult(
        status=status,
        mode=mode,
        decision=_decision(mode),
        proposal=None,
        execution_result=execution_result,
        message=message,
        validation_errors=validation_errors,
        confirmation_id=confirmation_id,
        executed=status is ExecutionCoordinationStatus.EXECUTED,
    )


def _tool_result(
    tool_name: str,
    result,
    *,
    action: str | None = None,
    arguments: dict[str, object] | None = None,
    success: bool = True,
    status: str = "success",
    executed: bool = True,
    error_message: str | None = None,
    confirmation_id: str | None = None,
) -> ToolRunResult:
    return ToolRunResult(
        success=success,
        status=status,
        intent=ToolIntent(action or tool_name, arguments or {}),
        tool_name=tool_name,
        original_arguments=arguments or {},
        validated_arguments=arguments or {},
        executed=executed,
        execution_count=1 if executed else 0,
        result=result,
        error_code=None if success else status,
        error_message=error_message,
        confirmation_id=confirmation_id,
    )


def _step(
    step_id: str,
    tool_name: str,
    result: ToolRunResult,
) -> ToolChainStepResult:
    return ToolChainStepResult(
        step_id=step_id,
        tool_name=tool_name,
        arguments=result.original_arguments or {},
        resolved_arguments=result.validated_arguments,
        result=result,
    )


def test_file_read_short_preserves_content_and_context() -> None:
    tool = _tool_result(
        "read_file",
        "línea 1\nlínea 2",
        action="file.read",
        arguments={"path": "README.md"},
    )

    text = ExecutionResultPresenter().present(
        _coordination(ExecutionCoordinationStatus.EXECUTED, tool)
    )

    assert text == "He leído README.md:\nlínea 1\nlínea 2"
    assert "ToolRunResult" not in text


def test_file_read_long_truncates_only_presentation() -> None:
    content = "á" * 30
    tool = _tool_result(
        "read_file",
        content,
        action="file.read",
        arguments={"path": "largo.txt"},
    )
    presenter = ExecutionResultPresenter(
        limits=PresentationLimits(short_text_limit=10, preview_character_limit=12)
    )

    text = presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, tool))

    assert "Mostrando una vista parcial" in text
    assert "12 de 30 caracteres" in text
    assert tool.result == content


def test_file_write_and_directory_list_present_clean_text() -> None:
    write = _tool_result(
        "write_file",
        "ok",
        action="file.write",
        arguments={"path": "prueba.txt", "content": "hola"},
    )
    listed = _tool_result(
        "list_directory",
        ["a.py", "docs"],
        action="directory.list",
        arguments={"path": "tools"},
    )

    presenter = ExecutionResultPresenter()

    assert presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, write)).startswith(
        "Listo. Escribí el archivo prueba.txt."
    )
    assert presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, listed)) == (
        "Contenido de tools:\n- a.py\n- docs"
    )


def test_empty_directory_and_project_tree_are_readable() -> None:
    empty = _tool_result(
        "list_directory",
        [],
        action="directory.list",
        arguments={"path": "vacía"},
    )
    tree = _tool_result(
        "project_tree",
        ["main.py", "tools/base_tool.py"],
        action="project.tree",
        arguments={"path": "."},
    )

    presenter = ExecutionResultPresenter()

    assert presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, empty)) == (
        "La carpeta vacía está vacía."
    )
    assert "Arbol del proyecto en ." in presenter.present(
        _coordination(ExecutionCoordinationStatus.EXECUTED, tree)
    )


def test_desktop_tools_present_action_context_without_long_text_dump() -> None:
    opened = _tool_result(
        "desktop.open_application",
        "VS Code abierto. PID: 1",
        action="desktop.application.open",
        arguments={"application": "VS Code"},
    )
    typed = _tool_result(
        "desktop.type_text",
        "Texto escrito.",
        action="desktop.text.type",
        arguments={"window_title": "Bloc", "text": "x" * 1000},
    )
    hotkey = _tool_result(
        "desktop.press_hotkey",
        "Atajo enviado.",
        action="desktop.hotkey.press",
        arguments={"window_title": "VS Code", "keys": ["ctrl", "s"]},
    )

    presenter = ExecutionResultPresenter(
        limits=PresentationLimits(preview_character_limit=20)
    )

    assert presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, opened)) == "Abrí VS Code."
    typed_text = presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, typed))
    assert "Texto escrito en Bloc." in typed_text
    assert "no se muestran 980 caracteres" in typed_text
    assert presenter.present(_coordination(ExecutionCoordinationStatus.EXECUTED, hotkey)) == (
        "Ejecuté Ctrl+S en VS Code."
    )


def test_unknown_tool_fallback_never_dumps_repr() -> None:
    unknown = _tool_result(
        "demo.unknown",
        {"value": 1},
        action="demo.unknown",
        arguments={"x": 1},
    )

    text = ExecutionResultPresenter().present(
        _coordination(ExecutionCoordinationStatus.EXECUTED, unknown)
    )

    assert text == "Operacion completada. Resultado: {value}"
    assert "ToolRunResult" not in text


def test_successful_read_write_chain_uses_natural_goal() -> None:
    read = _tool_result(
        "read_file",
        "contenido",
        action="file.read",
        arguments={"path": "README.md"},
    )
    write = _tool_result(
        "write_file",
        "ok",
        action="file.write",
        arguments={"path": "resumen.txt", "content": "contenido"},
    )
    chain = ToolChainResult(
        success=True,
        status="success",
        steps=(_step("read", "file.read", read), _step("write", "file.write", write)),
        execution_count=2,
    )

    text = ExecutionResultPresenter().present(
        _coordination(
            ExecutionCoordinationStatus.EXECUTED,
            chain,
            mode=ExecutionMode.TOOL_CHAIN,
        )
    )

    assert text == "Listo. Leí README.md y guardé su contenido en resumen.txt."
    assert "Cadena completada" not in text
    assert "read:" not in text


def test_three_step_chain_uses_final_result_without_repeating_large_content() -> None:
    first = _tool_result("demo.first", "a" * 500, action="demo.first")
    second = _tool_result("demo.second", "b" * 500, action="demo.second")
    third = _tool_result("demo.third", "final", action="demo.third")
    chain = ToolChainResult(
        success=True,
        status="success",
        steps=(
            _step("first", "demo.first", first),
            _step("second", "demo.second", second),
            _step("third", "demo.third", third),
        ),
        execution_count=3,
    )

    text = ExecutionResultPresenter(
        limits=PresentationLimits(preview_character_limit=20)
    ).present(
        _coordination(
            ExecutionCoordinationStatus.EXECUTED,
            chain,
            mode=ExecutionMode.TOOL_CHAIN,
        )
    )

    assert text == "Listo. Complete la cadena de acciones.\nOperacion completada.\nfinal"
    assert "a" * 100 not in text
    assert "b" * 100 not in text


def test_chain_failure_names_completed_and_failed_steps() -> None:
    read = _tool_result(
        "read_file",
        "contenido",
        action="file.read",
        arguments={"path": "README.md"},
    )
    write = _tool_result(
        "write_file",
        None,
        action="file.write",
        arguments={"path": "resumen.txt", "content": "contenido"},
        success=False,
        status="tool_execution_error",
        executed=True,
        error_message="permiso denegado",
    )
    chain = ToolChainResult(
        success=False,
        status="tool_execution_error",
        steps=(_step("read", "file.read", read), _step("write", "file.write", write)),
        failed_step_id="write",
        execution_count=2,
        error_message="permiso denegado",
    )

    text = ExecutionResultPresenter().present(
        _coordination(
            ExecutionCoordinationStatus.FAILED,
            chain,
            mode=ExecutionMode.TOOL_CHAIN,
        )
    )

    assert text == "Leí README.md, pero no pude escribir resumen.txt: permiso denegado."


def test_chain_cancelled_after_previous_step_does_not_claim_completion() -> None:
    read = _tool_result(
        "read_file",
        "contenido",
        action="file.read",
        arguments={"path": "README.md"},
    )
    write = _tool_result(
        "write_file",
        None,
        action="file.write",
        arguments={"path": "resumen.txt", "content": "contenido"},
        success=False,
        status="cancelled",
        executed=False,
        error_message="operation cancelled by user",
    )
    chain = ToolChainResult(
        success=False,
        status="cancelled",
        steps=(_step("read", "file.read", read), _step("write", "file.write", write)),
        failed_step_id="write",
        execution_count=1,
        error_message="operation cancelled by user",
    )

    text = ExecutionResultPresenter().present(
        _coordination(
            ExecutionCoordinationStatus.CANCELLED,
            chain,
            mode=ExecutionMode.TOOL_CHAIN,
        )
    )

    assert "Operacion cancelada." in text
    assert "paso pendiente no se ejecuto" in text


def test_statuses_and_debug_mode_hide_ids_in_normal_mode() -> None:
    pending = _tool_result(
        "write_file",
        None,
        action="file.write",
        arguments={"path": "salida.txt", "content": "hola"},
        success=False,
        status="confirmation_required",
        executed=False,
        confirmation_id="secret-id",
    )
    result = _coordination(
        ExecutionCoordinationStatus.CONFIRMATION_REQUIRED,
        pending,
        confirmation_id="secret-id",
    )
    presenter = ExecutionResultPresenter()

    normal = presenter.present(result)
    debug = presenter.present(result, debug=True)

    assert "secret-id" not in normal
    assert "Voy a escribir 'hola' en salida.txt." in normal
    assert "Detalles tecnicos:" in debug
    assert "execution_count" in debug
    assert "secret-id" not in debug


def test_failed_validation_and_unsupported_are_plain_text() -> None:
    presenter = ExecutionResultPresenter()

    validation = presenter.present(
        _coordination(
            ExecutionCoordinationStatus.VALIDATION_FAILED,
            validation_errors=("path: required argument is missing",),
        )
    )
    unsupported = presenter.present(
        _coordination(ExecutionCoordinationStatus.UNSUPPORTED, message="delete missing")
    )

    assert "path: required argument is missing" in validation
    assert unsupported == "Atlas todavia no dispone de la capacidad necesaria para esa accion."


def test_calendar_events_present_readable_text_without_internal_structure() -> None:
    tool = _tool_result(
        "calendar_list_events",
        {
            "events": [
                {
                    "id": "evt_1",
                    "summary": "Planning",
                    "start": "2026-08-09T10:00:00+01:00",
                    "end": "2026-08-09T10:30:00+01:00",
                }
            ]
        },
        action="calendar.events.list",
    )

    text = ExecutionResultPresenter().present(
        _coordination(ExecutionCoordinationStatus.EXECUTED, tool)
    )

    assert text == (
        "Eventos encontrados:\n"
        "- Planning: 2026-08-09T10:00:00+01:00 - 2026-08-09T10:30:00+01:00"
    )
    assert "evt_1" not in text
    assert "{events" not in text


def test_calendar_events_present_empty_result_clearly() -> None:
    tool = _tool_result(
        "calendar_list_events",
        {"events": []},
        action="calendar.events.list",
    )

    text = ExecutionResultPresenter().present(
        _coordination(ExecutionCoordinationStatus.EXECUTED, tool)
    )

    assert text == "No hay eventos en el rango solicitado."
    assert "{events" not in text
