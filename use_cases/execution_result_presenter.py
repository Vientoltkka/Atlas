"""Present structured execution results as concise user-facing text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tools.execution_coordinator import (
    ExecutionCoordinationResult,
    ExecutionCoordinationStatus,
)
from tools.execution_decision import ExecutionMode
from tools.single_tool_runner import ToolRunResult
from tools.tool_chain_proposal_builder import (
    StructuredToolChainProposal,
    StructuredToolChainStepProposal,
)
from tools.tool_chain_runner import ToolChainResult, ToolChainStepResult
from tools.tool_proposal_builder import StructuredToolProposal


@dataclass(frozen=True, slots=True)
class PresentationLimits:
    """Length limits applied only to displayed text."""

    short_text_limit: int = 1200
    preview_character_limit: int = 800
    max_list_items: int = 80
    max_tree_lines: int = 120


@dataclass(frozen=True, slots=True)
class PresentationContext:
    """Structured input for execution-result presentation."""

    original_text: str
    mode: ExecutionMode
    proposal: StructuredToolProposal | StructuredToolChainProposal | None
    execution_result: ToolRunResult | ToolChainResult | None


@dataclass(frozen=True, slots=True)
class PresentationResult:
    """Structured presentation output for user-facing and debug callers."""

    text: str
    summary: str
    details: tuple[str, ...] = ()
    technical_details: Mapping[str, object] | None = None


class ExecutionResultPresenter:
    """Convert structured execution results into deterministic console text."""

    def __init__(
        self,
        *,
        limits: PresentationLimits | None = None,
        debug: bool = False,
    ) -> None:
        self._limits = limits or PresentationLimits()
        self._debug = debug

    def present(
        self,
        result: ExecutionCoordinationResult,
        *,
        original_text: str | None = None,
        debug: bool | None = None,
    ) -> str:
        """Return user-facing text without dumping internal structures."""
        presentation = self.present_structured(
            result,
            original_text=original_text,
            debug=debug,
        )
        return presentation.text

    def present_structured(
        self,
        result: ExecutionCoordinationResult,
        *,
        original_text: str | None = None,
        debug: bool | None = None,
    ) -> PresentationResult:
        """Return structured presentation data for tests and future callers."""
        context = PresentationContext(
            original_text=original_text or _proposal_source(result.proposal),
            mode=result.mode,
            proposal=result.proposal,
            execution_result=result.execution_result,
        )
        show_debug = self._debug if debug is None else debug

        try:
            presentation = self._present_result(result, context)
        except Exception as error:
            presentation = PresentationResult(
                text="Operacion completada, pero no he podido formatear todos los detalles.",
                summary="presentation fallback",
                details=(str(error),),
            )

        if show_debug:
            return _with_debug_details(presentation, result)

        return presentation

    def _confirmation_message(
        self,
        result: ExecutionCoordinationResult,
    ) -> str:
        return self._present_confirmation(result).text

    def _present_result(
        self,
        result: ExecutionCoordinationResult,
        context: PresentationContext,
    ) -> PresentationResult:
        if result.status is ExecutionCoordinationStatus.EXECUTED:
            return self._present_executed(context)

        if result.status is ExecutionCoordinationStatus.INFORMATION_REQUIRED:
            return PresentationResult(
                text=_missing_information_message(result.missing_information),
                summary="missing information",
                details=result.missing_information,
            )

        if result.status is ExecutionCoordinationStatus.AMBIGUOUS_REQUEST:
            return PresentationResult(
                text=_ambiguous_information_message(result.ambiguous_information),
                summary="ambiguous request",
                details=result.ambiguous_information,
            )

        if result.status is ExecutionCoordinationStatus.UNSUPPORTED:
            return PresentationResult(
                text="Atlas todavia no dispone de la capacidad necesaria para esa accion.",
                summary="unsupported",
                details=(result.message,),
            )

        if result.status is ExecutionCoordinationStatus.VALIDATION_FAILED:
            message = _validation_failed_message(result.validation_errors)
            return PresentationResult(
                text=message,
                summary="validation failed",
                details=result.validation_errors,
            )

        if result.status is ExecutionCoordinationStatus.CONFIRMATION_REQUIRED:
            return self._present_confirmation(result)

        if result.status is ExecutionCoordinationStatus.CANCELLED:
            return self._present_cancelled(context)

        if result.status is ExecutionCoordinationStatus.FAILED:
            if isinstance(context.execution_result, ToolChainResult):
                return self._present_chain(context.execution_result, context)
            return PresentationResult(
                text=_failure_message(result.execution_result, result.message),
                summary="failed",
                details=(result.message,),
            )

        return PresentationResult(
            text=result.message,
            summary="message",
            details=(result.message,),
        )

    def _present_executed(
        self,
        context: PresentationContext,
    ) -> PresentationResult:
        execution_result = context.execution_result

        if isinstance(execution_result, ToolChainResult):
            return self._present_chain(execution_result, context)

        if isinstance(execution_result, ToolRunResult):
            return self._present_tool_result(execution_result)

        return PresentationResult(
            text="Operacion completada.",
            summary="completed",
        )

    def _present_tool_result(
        self,
        result: ToolRunResult,
        *,
        allow_large_output: bool = True,
    ) -> PresentationResult:
        if not result.success:
            return PresentationResult(
                text=_tool_failure_message(result),
                summary="tool failed",
                details=_compact_details(result),
            )

        if result.tool_name == "read_file":
            return self._present_read_file(result, allow_large_output=allow_large_output)

        if result.tool_name == "write_file":
            return self._present_write_file(result)

        if result.tool_name == "list_directory":
            return self._present_directory_list(result)

        if result.tool_name == "project_tree":
            return self._present_project_tree(result)

        if result.tool_name == "desktop.open_application":
            return self._present_open_application(result)

        if result.tool_name == "desktop.type_text":
            return self._present_type_text(result)

        if result.tool_name == "desktop.press_hotkey":
            return self._present_hotkey(result)

        if result.tool_name and result.tool_name.startswith("desktop."):
            return self._present_desktop_fallback(result)

        return self._present_generic_tool(result)

    def _present_read_file(
        self,
        result: ToolRunResult,
        *,
        allow_large_output: bool,
    ) -> PresentationResult:
        path = _argument(result, "path") or "el archivo solicitado"
        content = "" if result.result is None else str(result.result)
        rendered = (
            self._content_block(content)
            if allow_large_output
            else self._preview_text(content, label="contenido")
        )
        text = f"He leído {path}:\n{rendered}"
        return PresentationResult(
            text=text,
            summary=f"read {path}",
            details=(path,),
        )

    def _present_write_file(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        path = _argument(result, "path")
        content = _argument(result, "content")

        if path:
            lines = [f"Listo. Escribí el archivo {path}."]
        else:
            lines = ["Listo. Escribí el archivo solicitado."]

        if isinstance(content, str) and len(content) <= 80 and "\n" not in content:
            lines.append(f"Contenido escrito: {content}")

        return PresentationResult(
            text="\n".join(lines),
            summary="file written",
            details=(str(path),) if path else (),
        )

    def _present_directory_list(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        path = _argument(result, "path") or "."
        items = _as_sequence(result.result)

        if not items:
            return PresentationResult(
                text=f"La carpeta {path} está vacía.",
                summary="directory empty",
                details=(str(path),),
            )

        body = self._format_list(items, limit=self._limits.max_list_items)
        return PresentationResult(
            text=f"Contenido de {path}:\n{body}",
            summary=f"listed {path}",
            details=(str(path),),
        )

    def _present_project_tree(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        path = _argument(result, "path") or "."
        items = _as_sequence(result.result)

        if not items:
            return PresentationResult(
                text=f"No hay archivos Python dentro de {path}.",
                summary="project tree empty",
                details=(str(path),),
            )

        body = self._format_list(items, limit=self._limits.max_tree_lines)
        return PresentationResult(
            text=f"Arbol del proyecto en {path}:\n{body}",
            summary=f"project tree {path}",
            details=(str(path),),
        )

    def _present_open_application(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        application = _argument(result, "application") or "la aplicacion"
        return PresentationResult(
            text=f"Abrí {application}.",
            summary="application opened",
            details=(str(application),),
        )

    def _present_type_text(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        title = _argument(result, "window_title")
        typed = _argument(result, "text")
        preview = self._preview_text(str(typed), label="texto") if typed is not None else ""
        destination = f" en {title}" if title else ""
        lines = [f"Texto escrito{destination}."]
        if preview:
            lines.append(preview)
        return PresentationResult(
            text="\n".join(lines),
            summary="text typed",
            details=(str(title),) if title else (),
        )

    def _present_hotkey(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        keys = _format_hotkey(_argument(result, "keys"))
        title = _argument(result, "window_title")
        if title:
            text = f"Ejecuté {keys} en {title}."
        else:
            text = f"Ejecuté {keys}."
        return PresentationResult(
            text=text,
            summary="hotkey pressed",
            details=(keys,),
        )

    def _present_desktop_fallback(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        message = str(result.result) if result.result is not None else "Accion de escritorio completada."
        return PresentationResult(
            text=message,
            summary="desktop action",
            details=_compact_details(result),
        )

    def _present_generic_tool(
        self,
        result: ToolRunResult,
    ) -> PresentationResult:
        if isinstance(result.result, str):
            rendered = self._preview_text(result.result, label="resultado")
            return PresentationResult(
                text=f"Operacion completada.\n{rendered}",
                summary="generic completed",
                details=_compact_details(result),
            )

        if isinstance(result.result, bool):
            value = "si" if result.result else "no"
            return PresentationResult(
                text=f"Operacion completada. Resultado: {value}.",
                summary="generic completed",
                details=(value,),
            )

        if isinstance(result.result, list | tuple):
            body = self._format_list(result.result, limit=self._limits.max_list_items)
            return PresentationResult(
                text=f"Operacion completada:\n{body}",
                summary="generic completed",
                details=_compact_details(result),
            )

        if result.result is None:
            return PresentationResult(
                text="Operacion completada.",
                summary="generic completed",
            )

        return PresentationResult(
            text=f"Operacion completada. Resultado: {_safe_scalar(result.result)}",
            summary="generic completed",
            details=_compact_details(result),
        )

    def _present_chain(
        self,
        chain: ToolChainResult,
        context: PresentationContext,
    ) -> PresentationResult:
        if chain.success:
            text = self._successful_chain_text(chain)
            return PresentationResult(
                text=text,
                summary="chain completed",
                details=_chain_step_summaries(chain),
            )

        if chain.status == "cancelled":
            return PresentationResult(
                text=self._cancelled_chain_text(chain),
                summary="chain cancelled",
                details=_chain_step_summaries(chain),
            )

        return PresentationResult(
            text=self._failed_chain_text(chain),
            summary="chain failed",
            details=_chain_step_summaries(chain),
        )

    def _successful_chain_text(
        self,
        chain: ToolChainResult,
    ) -> str:
        if _is_read_write_chain(chain):
            source = _argument(chain.steps[0].result, "path") or "el archivo origen"
            target = _argument(chain.steps[-1].result, "path") or "el archivo destino"
            return f"Listo. Leí {source} y guardé su contenido en {target}."

        lines = ["Listo. Complete la cadena de acciones."]
        final_step = chain.steps[-1] if chain.steps else None
        if final_step is not None:
            final = self._present_tool_result(final_step.result, allow_large_output=False).text
            lines.append(final)
        return "\n".join(lines)

    def _cancelled_chain_text(
        self,
        chain: ToolChainResult,
    ) -> str:
        completed = _completed_steps(chain)
        pending = _failed_step(chain)
        if completed and pending is not None:
            completed_text = _natural_completed_phrase(completed)
            return (
                f"Operacion cancelada. {completed_text}, pero el paso pendiente "
                f"no se ejecuto."
            )
        return "Operacion cancelada."

    def _failed_chain_text(
        self,
        chain: ToolChainResult,
    ) -> str:
        completed = _completed_steps(chain)
        failed = _failed_step(chain)
        reason = chain.error_message or "no se pudo completar el paso"

        if _is_read_write_failure(chain):
            source = _argument(completed[0].result, "path")
            target = _argument(failed.result, "path") if failed is not None else None
            if source and target:
                return f"Leí {source}, pero no pude escribir {target}: {reason}."

        if failed is None:
            return f"No he podido completar la cadena: {reason}."

        completed_text = ""
        if completed:
            completed_text = _natural_completed_phrase(completed) + ", pero "

        return (
            f"{completed_text}no pude completar el paso "
            f"{_tool_action_label(failed.result.tool_name)}: {reason}."
        )

    def _present_confirmation(
        self,
        result: ExecutionCoordinationResult,
    ) -> PresentationResult:
        execution_result = result.execution_result
        pending = _pending_tool_result(execution_result)

        if _is_ambiguous_confirmation(execution_result):
            text = (
                "No he entendido la confirmacion. Responde si/s para continuar "
                "o no/n para cancelar."
            )
            return PresentationResult(text=text, summary="ambiguous confirmation")

        if isinstance(execution_result, ToolChainResult):
            message = self._pending_chain_message(execution_result)
        else:
            message = self._pending_tool_message(pending)

        return PresentationResult(
            text=f"{message}\nDeseas continuar? [s/N]",
            summary="confirmation required",
            details=(message,),
        )

    def _pending_chain_message(
        self,
        chain: ToolChainResult,
    ) -> str:
        completed = _completed_steps(chain)
        pending = _failed_step(chain) or (chain.steps[-1] if chain.steps else None)

        if completed and pending is not None:
            source = _argument(completed[-1].result, "path")
            target = _argument(pending.result, "path")
            if source and target:
                return f"Ya leí {source}. El siguiente paso escribirá ese contenido en {target}."

        if pending is not None:
            return self._pending_tool_message(pending.result)

        return "Esta operacion requiere confirmacion."

    def _pending_tool_message(
        self,
        result: ToolRunResult | None,
    ) -> str:
        if result is None:
            return "Esta operacion requiere confirmacion."

        path = _argument(result, "path")
        content = _argument(result, "content")

        if result.tool_name == "write_file" and path:
            if isinstance(content, str):
                return f"Voy a escribir {_content_summary(content)} en {path}."
            return f"Voy a escribir el archivo {path}."

        if result.tool_name == "desktop.type_text":
            title = _argument(result, "window_title")
            typed = _argument(result, "text")
            destination = f" en {title}" if title else ""
            return f"Voy a escribir {_content_summary(str(typed))}{destination}."

        if result.tool_name == "desktop.press_hotkey":
            keys = _format_hotkey(_argument(result, "keys"))
            title = _argument(result, "window_title")
            if title:
                return f"Voy a ejecutar {keys} en {title}."
            return f"Voy a ejecutar {keys}."

        if path:
            return f"Esta operacion requiere confirmacion: {_tool_action_label(result.tool_name)} sobre {path}."

        return f"Esta operacion requiere confirmacion: {_tool_action_label(result.tool_name)}."

    def _present_cancelled(
        self,
        context: PresentationContext,
    ) -> PresentationResult:
        if isinstance(context.execution_result, ToolChainResult):
            return PresentationResult(
                text=self._cancelled_chain_text(context.execution_result),
                summary="cancelled",
            )

        return PresentationResult(
            text="Operacion cancelada.",
            summary="cancelled",
        )

    def _content_block(
        self,
        content: str,
    ) -> str:
        if len(content) <= self._limits.short_text_limit:
            return content

        preview = content[: self._limits.preview_character_limit]
        omitted = len(content) - len(preview)
        line_count = content.count("\n") + 1 if content else 0
        return (
            f"{preview}\n"
            f"[Mostrando una vista parcial: {len(preview)} de {len(content)} "
            f"caracteres, {line_count} lineas. No se muestran {omitted} caracteres.]"
        )

    def _preview_text(
        self,
        content: str,
        *,
        label: str,
    ) -> str:
        if len(content) <= self._limits.preview_character_limit:
            return content

        preview = content[: self._limits.preview_character_limit]
        omitted = len(content) - len(preview)
        return (
            f"{label.capitalize()} parcial:\n{preview}\n"
            f"[Mostrando una vista parcial: no se muestran {omitted} caracteres.]"
        )

    def _format_list(
        self,
        items: tuple[Any, ...],
        *,
        limit: int,
    ) -> str:
        visible = items[:limit]
        lines = [f"- {_safe_scalar(item)}" for item in visible]
        omitted = len(items) - len(visible)
        if omitted > 0:
            lines.append(f"- [Mostrando una vista parcial: no se muestran {omitted} elementos.]")
        return "\n".join(lines)


def _proposal_source(
    proposal: StructuredToolProposal | StructuredToolChainProposal | None,
) -> str:
    if proposal is None:
        return ""
    return proposal.source_text


def _with_debug_details(
    presentation: PresentationResult,
    result: ExecutionCoordinationResult,
) -> PresentationResult:
    technical = {
        "mode": result.mode.value,
        "status": result.status.value,
        "executed": result.executed,
        "confirmation_pending": result.confirmation_id is not None,
    }
    execution_result = result.execution_result
    if isinstance(execution_result, ToolRunResult):
        technical.update(
            {
                "tool_name": execution_result.tool_name,
                "runner_status": execution_result.status,
                "execution_count": execution_result.execution_count,
                "error_code": execution_result.error_code,
            }
        )
    elif isinstance(execution_result, ToolChainResult):
        technical.update(
            {
                "runner_status": execution_result.status,
                "execution_count": execution_result.execution_count,
                "failed_step_id": execution_result.failed_step_id,
                "step_ids": tuple(step.step_id for step in execution_result.steps),
            }
        )

    lines = [presentation.text, "", "Detalles tecnicos:"]
    for key, value in technical.items():
        lines.append(f"- {key}: {value}")

    return PresentationResult(
        text="\n".join(lines),
        summary=presentation.summary,
        details=presentation.details,
        technical_details=technical,
    )


def _missing_information_message(
    fields: tuple[str, ...],
) -> str:
    if not fields:
        return "Necesito mas informacion para poder ejecutar esa accion."
    return "Necesito que indiques: " + ", ".join(fields) + "."


def _ambiguous_information_message(
    fields: tuple[str, ...],
) -> str:
    if not fields:
        return "La peticion es ambigua. Aclara los datos antes de ejecutar."
    return "Necesito que aclares: " + ", ".join(fields) + "."


def _validation_failed_message(
    errors: tuple[str, ...],
) -> str:
    if not errors:
        return "La peticion no ha pasado la validacion."
    return "La peticion no ha pasado la validacion: " + "; ".join(errors)


def _failure_message(
    execution_result: ToolRunResult | ToolChainResult | None,
    fallback: str,
) -> str:
    if isinstance(execution_result, ToolRunResult):
        return _tool_failure_message(execution_result)
    if isinstance(execution_result, ToolChainResult) and execution_result.error_message:
        return f"No he podido completar la operacion: {execution_result.error_message}."
    if fallback:
        return f"No he podido completar la operacion: {fallback}."
    return "No he podido completar la operacion."


def _tool_failure_message(
    result: ToolRunResult,
) -> str:
    reason = result.error_message or "no se pudo completar"
    path = _argument(result, "path")
    if result.tool_name == "read_file" and path:
        return f"No pude leer {path}: {reason}."
    if result.tool_name == "write_file" and path:
        return f"No pude escribir {path}: {reason}."
    if result.tool_name and result.tool_name.startswith("desktop."):
        return f"No pude completar la accion de escritorio: {reason}."
    return f"No he podido completar la operacion: {reason}."


def _pending_tool_result(
    execution_result: ToolRunResult | ToolChainResult | None,
) -> ToolRunResult | None:
    if isinstance(execution_result, ToolRunResult):
        return execution_result
    if isinstance(execution_result, ToolChainResult) and execution_result.steps:
        return execution_result.steps[-1].result
    return None


def _is_ambiguous_confirmation(
    execution_result: ToolRunResult | ToolChainResult | None,
) -> bool:
    if isinstance(execution_result, ToolRunResult):
        return execution_result.status == "invalid_confirmation"
    if isinstance(execution_result, ToolChainResult):
        return execution_result.status == "invalid_confirmation"
    return False


def _argument(
    result: ToolRunResult,
    name: str,
) -> Any:
    if result.validated_arguments and name in result.validated_arguments:
        return result.validated_arguments[name]
    if result.original_arguments and name in result.original_arguments:
        return result.original_arguments[name]
    return None


def _as_sequence(
    value: Any,
) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _safe_scalar(
    value: Any,
) -> str:
    if isinstance(value, Mapping):
        keys = ", ".join(str(key) for key in list(value.keys())[:6])
        return "{" + keys + "}"
    return str(value)


def _compact_details(
    result: ToolRunResult,
) -> tuple[str, ...]:
    details = []
    if result.tool_name:
        details.append(result.tool_name)
    if result.status:
        details.append(result.status)
    return tuple(details)


def _format_hotkey(
    keys: Any,
) -> str:
    if isinstance(keys, list | tuple):
        return "+".join(str(key).upper() if len(str(key)) == 1 else str(key).capitalize() for key in keys)
    return str(keys or "el atajo solicitado")


def _content_summary(
    content: str,
) -> str:
    if len(content) <= 80 and "\n" not in content:
        return repr(content)
    line_count = content.count("\n") + 1 if content else 0
    return f"{len(content)} caracteres en {line_count} lineas"


def _tool_action_label(
    tool_name: str | None,
) -> str:
    labels = {
        "read_file": "leer archivo",
        "write_file": "escribir archivo",
        "list_directory": "listar directorio",
        "project_tree": "mostrar el arbol del proyecto",
        "desktop.open_application": "abrir aplicacion",
        "desktop.open_file": "abrir archivo",
        "desktop.type_text": "escribir texto",
        "desktop.press_hotkey": "ejecutar atajo",
    }
    if tool_name is None:
        return "accion"
    return labels.get(tool_name, tool_name.replace("_", "."))


def _completed_steps(
    chain: ToolChainResult,
) -> tuple[ToolChainStepResult, ...]:
    return tuple(step for step in chain.steps if step.result.success and step.result.executed)


def _failed_step(
    chain: ToolChainResult,
) -> ToolChainStepResult | None:
    if chain.failed_step_id is not None:
        for step in chain.steps:
            if step.step_id == chain.failed_step_id:
                return step
    for step in reversed(chain.steps):
        if not step.result.success:
            return step
    return None


def _chain_step_summaries(
    chain: ToolChainResult,
) -> tuple[str, ...]:
    return tuple(
        f"{step.step_id}:{step.tool_name}:{step.result.status}"
        for step in chain.steps
    )


def _is_read_write_chain(
    chain: ToolChainResult,
) -> bool:
    return (
        len(chain.steps) >= 2
        and chain.steps[0].tool_name == "file.read"
        and chain.steps[-1].tool_name == "file.write"
        and chain.steps[0].result.success
        and chain.steps[-1].result.success
    )


def _is_read_write_failure(
    chain: ToolChainResult,
) -> bool:
    completed = _completed_steps(chain)
    failed = _failed_step(chain)
    return (
        bool(completed)
        and completed[0].tool_name == "file.read"
        and failed is not None
        and failed.tool_name == "file.write"
    )


def _natural_completed_phrase(
    steps: tuple[ToolChainStepResult, ...],
) -> str:
    phrases = []
    for step in steps:
        path = _argument(step.result, "path")
        action = _tool_action_label(step.result.tool_name)
        if path:
            phrases.append(f"{action.capitalize()} {path}")
        else:
            phrases.append(action.capitalize())
    return ", ".join(phrases)
