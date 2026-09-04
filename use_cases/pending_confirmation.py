"""Resolve safe user input while an execution confirmation is pending."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any, Literal, Mapping
import unicodedata

from tools.execution_decision import ExecutionMode
from tools.single_tool_runner import ToolRunResult
from tools.tool_chain_runner import ToolChainResult, ToolChainStepResult
from tools.tool_chain_proposal_builder import StructuredToolChainProposal
from tools.tool_proposal_builder import StructuredToolProposal


ConfirmationOwner = Literal["single", "chain"]


class PendingConfirmationInputType(str, Enum):
    """Intent classes accepted while a confirmation is pending."""

    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    AMBIGUOUS = "AMBIGUOUS"
    INSPECT = "INSPECT"
    MODIFY = "MODIFY"
    REPLACE = "REPLACE"
    CANCEL = "CANCEL"


@dataclass(frozen=True, slots=True)
class PendingConfirmationContext:
    """Explicit state needed to describe or safely modify a pending operation."""

    confirmation_id: str
    mode: ExecutionMode
    owner: ConfirmationOwner
    original_text: str
    proposal: StructuredToolProposal | StructuredToolChainProposal | None
    execution_result: ToolRunResult | ToolChainResult | None
    action_summary: str


@dataclass(frozen=True, slots=True)
class PendingConfirmationResolution:
    """Classification and extracted data for one pending-confirmation turn."""

    input_type: PendingConfirmationInputType
    arguments: Mapping[str, Any] | None = None
    replacement_text: str | None = None
    blocked_reason: str | None = None


class PendingConfirmationResolver:
    """Deterministically classify and parse confirmation-time input."""

    def resolve(
        self,
        text: str,
        context: PendingConfirmationContext,
    ) -> PendingConfirmationResolution:
        """Return the confirmation-time intent before any execution can happen."""
        normalized = _normalize(text)

        if not normalized:
            return PendingConfirmationResolution(PendingConfirmationInputType.AMBIGUOUS)

        if _looks_like_replace(normalized):
            return PendingConfirmationResolution(
                PendingConfirmationInputType.REPLACE,
                replacement_text=_replacement_text(text),
            )

        if _is_confirm(normalized):
            return PendingConfirmationResolution(PendingConfirmationInputType.CONFIRM)

        if _is_reject(normalized):
            input_type = (
                PendingConfirmationInputType.CANCEL
                if _has_cancel_signal(normalized)
                else PendingConfirmationInputType.REJECT
            )
            return PendingConfirmationResolution(input_type)

        if _is_inspect(normalized):
            return PendingConfirmationResolution(PendingConfirmationInputType.INSPECT)

        pending = _pending_tool_result(context.execution_result)
        if pending is not None and _looks_like_modify(normalized, pending.tool_name):
            blocked = _blocked_chain_source_change(normalized, context)
            if blocked is not None:
                return PendingConfirmationResolution(
                    PendingConfirmationInputType.MODIFY,
                    blocked_reason=blocked,
                )

            arguments = _extract_modified_arguments(text, normalized, pending)
            if arguments:
                return PendingConfirmationResolution(
                    PendingConfirmationInputType.MODIFY,
                    arguments=arguments,
                )

        return PendingConfirmationResolution(PendingConfirmationInputType.AMBIGUOUS)


class PendingConfirmationPresenter:
    """Build natural descriptions of pending execution state."""

    def describe(
        self,
        context: PendingConfirmationContext,
        *,
        prefix: str | None = None,
        inspect: bool = False,
    ) -> str:
        """Describe the operation without exposing the confirmation id."""
        execution_result = context.execution_result
        if isinstance(execution_result, ToolChainResult):
            message = self._describe_chain(execution_result, inspect=inspect)
        else:
            message = self._describe_tool(_pending_tool_result(execution_result), inspect=inspect)

        if prefix:
            message = f"{prefix} {message[0].lower()}{message[1:]}"

        if inspect:
            return message

        return f"{message}\nDeseas continuar? [s/N]"

    def _describe_chain(
        self,
        result: ToolChainResult,
        *,
        inspect: bool,
    ) -> str:
        completed = [step for step in result.steps if step.result.executed]
        pending = result.steps[-1] if result.steps else None

        if inspect:
            lines = ["Operacion pendiente en cadena:"]
            for step in completed:
                lines.append(f"- Ejecutado: {step.step_id} ({_tool_label(step.tool_name)})")
                path = _argument_from_mapping(step.result.validated_arguments, "path")
                if path:
                    lines.append(f"  Ruta: {path}")
            if pending is not None:
                lines.append(f"- Pendiente: {pending.step_id} ({_tool_label(pending.tool_name)})")
                path = _argument_from_mapping(pending.result.validated_arguments, "path")
                if path:
                    lines.append(f"  Ruta: {path}")
                content = _argument_from_mapping(pending.result.validated_arguments, "content")
                if content is not None:
                    lines.append(f"  Contenido: {_summarize_content(content)}")
            return "\n".join(lines)

        if completed and pending is not None:
            source = _argument_from_mapping(completed[-1].result.validated_arguments, "path")
            target = _argument_from_mapping(pending.result.validated_arguments, "path")
            if source and target:
                return (
                    f"Ya lei {source}. El siguiente paso escribira ese contenido "
                    f"en {target}."
                )

        if pending is not None:
            return self._describe_tool(pending.result, inspect=False)

        return "Esta operacion requiere confirmacion."

    def _describe_tool(
        self,
        result: ToolRunResult | None,
        *,
        inspect: bool,
    ) -> str:
        if result is None:
            return "Esta operacion requiere confirmacion."

        tool = _tool_label(result.tool_name)
        path = _argument_from_mapping(result.validated_arguments, "path")
        content = _argument_from_mapping(result.validated_arguments, "content")

        if inspect:
            lines = [f"Operacion pendiente: {tool}."]
            if path:
                lines.append(f"Ruta afectada: {path}.")
            if content is not None:
                lines.append(f"Contenido: {_summarize_content(content)}.")
            return "\n".join(lines)

        if result.tool_name == "write_file" and path and content is not None:
            return f"Voy a escribir {_summarize_content(content)} en {path}."

        if path:
            return f"Esta operacion requiere confirmacion: {tool} sobre {path}."

        return f"Esta operacion requiere confirmacion: {tool}."


def context_with_new_result(
    context: PendingConfirmationContext,
    result: ToolRunResult | ToolChainResult,
    confirmation_id: str,
) -> PendingConfirmationContext:
    """Return a context copy bound to a newly issued confirmation id."""
    return replace(
        context,
        confirmation_id=confirmation_id,
        execution_result=result,
    )


def _pending_tool_result(
    execution_result: ToolRunResult | ToolChainResult | None,
) -> ToolRunResult | None:
    if isinstance(execution_result, ToolRunResult):
        return execution_result
    if isinstance(execution_result, ToolChainResult) and execution_result.steps:
        return execution_result.steps[-1].result
    return None


def _blocked_chain_source_change(
    normalized: str,
    context: PendingConfirmationContext,
) -> str | None:
    if context.owner != "chain" or not isinstance(context.execution_result, ToolChainResult):
        return None

    read_executed = any(
        step.tool_name == "file.read" and step.result.executed
        for step in context.execution_result.steps
    )
    if read_executed and re.search(r"\b(?:lee|leer|origen|fuente)\b", normalized):
        return (
            "La lectura de origen ya se ejecuto. Para cambiarla hay que cancelar "
            "esta cadena y crear una operacion nueva."
        )
    return None


def _extract_modified_arguments(
    text: str,
    normalized: str,
    pending: ToolRunResult,
) -> dict[str, Any]:
    tool_name = pending.tool_name
    arguments: dict[str, Any] = {}

    if tool_name == "write_file":
        path = _extract_path(text)
        content = _extract_write_content(text, normalized, path)
        if path is not None:
            arguments["path"] = path
        if content is not None:
            arguments["content"] = content
        return arguments

    if tool_name in {"read_file", "desktop.open_file"}:
        path = _extract_path(text)
        return {"path": path} if path is not None else {}

    if tool_name == "desktop.open_application":
        application = _extract_application(text, normalized)
        return {"application": application} if application is not None else {}

    if tool_name == "desktop.press_hotkey":
        keys = _extract_hotkey(normalized)
        return {"keys": keys} if keys else {}

    if tool_name == "desktop.type_text":
        typed = _extract_type_text(text, normalized)
        return {"text": typed} if typed is not None else {}

    return {}


def _is_confirm(normalized: str) -> bool:
    return normalized in {
        "s",
        "si",
        "yes",
        "y",
        "confirma",
        "confirmo",
        "confirmar",
        "vale",
        "ok",
        "adelante",
        "continua",
        "continuar",
    }


def _is_reject(normalized: str) -> bool:
    return normalized in {"n", "no", "cancela", "cancelar", "olvidalo", "descarta", "rechaza"}


def _has_cancel_signal(normalized: str) -> bool:
    return bool(re.search(r"\b(?:cancela|cancelar|olvidalo|descarta)\b", normalized))


def _is_inspect(normalized: str) -> bool:
    has_question = re.search(r"\b(?:que|cual|muestra|explica)\b", normalized)
    has_object = re.search(r"\b(?:operacion|vas a hacer|archivo|accion|ejecutar|modificar)\b", normalized)
    return bool(has_question and has_object)


def _looks_like_modify(normalized: str, tool_name: str | None) -> bool:
    if re.search(r"\b(?:mejor|cambia|usa|en vez de|guardalo en|guardala en|destino|contenido|ventana|teclas)\b", normalized):
        return True
    if tool_name == "write_file" and re.search(r"\b(?:escribe|guarda|copia)\b", normalized):
        return True
    if tool_name == "desktop.press_hotkey" and re.search(r"\b(?:pulsa|atajo|ctrl|alt|shift)\b", normalized):
        return True
    return False


def _looks_like_replace(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:cancela|cancelar|descarta|sustituye|reemplaza)\b", normalized)
        and re.search(r"\b(?: y | luego | despues | lista|lee|escribe|abre|pulsa)\b", f" {normalized} ")
    )


def _replacement_text(text: str) -> str:
    match = re.search(
        r"\b(?:cancela|cancelar|descarta|sustituye|reemplaza)(?:\s+eso|\s+lo|\s+la operacion|\s+esta operacion)?\s+(?:y|por|con)?\s*(?P<rest>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("rest").strip()
    return text.strip()


def _extract_path(text: str) -> str | None:
    quoted = _extract_quoted(text)
    if quoted and _looks_like_path(quoted):
        return quoted

    extension_match = re.search(
        r"(?P<path>(?:[A-Za-z]:[\\/])?(?:[\w .-]+[\\/])*[\w .-]+\.(?:md|txt|py|json|csv|yaml|yml|toml))\b",
        text,
        flags=re.IGNORECASE,
    )
    if extension_match:
        return extension_match.group("path").strip()

    return None


def _extract_write_content(
    text: str,
    normalized: str,
    path: str | None,
) -> str | None:
    quoted = _extract_quoted(text)
    if quoted and quoted != path and not _looks_like_path(quoted):
        return quoted

    match = re.search(
        r"\b(?:mejor\s+)?(?:escribe|guarda|copia)\s+(?P<content>.+?)(?:\s+\ben\b|\s+\ba\b)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group("content").strip()
        if content and not _looks_like_path(content):
            return content

    match = re.search(
        r"\bcontenido\s+(?:a|por|sea)?\s*(?P<content>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group("content").strip()
        if content:
            return content

    if path is None and normalized.startswith("escribe "):
        content = text.split(None, 1)[1].strip()
        if content:
            return content

    return None


def _extract_application(text: str, normalized: str) -> str | None:
    aliases = {
        "visual studio code": "VS Code",
        "vs code": "VS Code",
        "vscode": "VS Code",
        "bloc de notas": "notepad",
        "notepad": "notepad",
    }
    for alias, application in aliases.items():
        if alias in normalized:
            return application

    quoted = _extract_quoted(text)
    if quoted:
        return quoted

    return None


def _extract_hotkey(normalized: str) -> list[str] | None:
    match = re.search(
        r"\b(?P<keys>(?:ctrl|control|alt|shift|mayus|win|windows)(?:\s*[+]\s*|\s+mas\s+|\s+)(?:[\w]+)(?:(?:\s*[+]\s*|\s+mas\s+|\s+)(?:[\w]+))*)\b",
        normalized,
    )
    if not match:
        return None
    return [_normalize_key(key) for key in re.split(r"\s*(?:[+]|\bmas\b)\s*|\s+", match.group("keys")) if key.strip()]


def _extract_type_text(text: str, normalized: str) -> str | None:
    quoted = _extract_quoted(text)
    if quoted:
        return quoted
    match = re.search(r"\b(?:escribe|teclea)\s+(?P<text>.+)$", text, flags=re.IGNORECASE)
    if match and "archivo" not in normalized:
        return match.group("text").strip()
    return None


def _extract_quoted(text: str) -> str | None:
    match = re.search(r"[\"'\u201c\u201d\u2018\u2019](?P<value>.+?)[\"'\u201c\u201d\u2018\u2019]", text)
    if not match:
        return None
    value = match.group("value").strip()
    return value or None


def _looks_like_path(value: str) -> bool:
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}\b", value) or "/" in value or "\\" in value)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    without_punctuation = re.sub(r"[^\w\s./+:-]", " ", without_accents)
    return " ".join(without_punctuation.split())


def _normalize_key(key: str) -> str:
    aliases = {
        "control": "ctrl",
        "mayus": "shift",
        "windows": "win",
        "escape": "esc",
        "intro": "enter",
    }
    return aliases.get(key.strip().lower(), key.strip().lower())


def _tool_label(tool_name: str | None) -> str:
    labels = {
        "read_file": "leer archivo",
        "write_file": "escribir archivo",
        "list_directory": "listar directorio",
        "desktop.open_application": "abrir aplicacion",
        "desktop.open_file": "abrir archivo",
        "desktop.press_hotkey": "pulsar teclas",
        "desktop.type_text": "escribir texto",
    }
    if tool_name is None:
        return "accion"
    return labels.get(tool_name, tool_name.replace("_", "."))


def _argument_from_mapping(arguments: Mapping[str, Any] | None, name: str) -> Any:
    if arguments and name in arguments:
        return arguments[name]
    return None


def _summarize_content(content: Any) -> str:
    text = str(content)
    if len(text) <= 80:
        return repr(text)
    return f"{len(text)} caracteres"
