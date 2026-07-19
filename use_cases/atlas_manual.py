"""Load and validate the internal Atlas manual without side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Mapping

from bootstrap.bootstrap import Bootstrap
from tools.argument_schema import ArgumentSchemaRegistry
from tools.intent_selector import ToolIntent, ToolSelector
from tools.registry import ToolRegistry


MANUAL_VERSION = "5.5A"
MANUAL_ROOT = Path("docs") / "manual"
KNOWN_CAPABILITY_STATES = {"IMPLEMENTED", "PARTIAL", "PLANNED", "UNSUPPORTED"}
REQUIRED_SECTION_IDS = (
    "overview",
    "architecture",
    "capabilities",
    "tools",
    "execution_flow",
    "confirmations",
    "conversation",
    "operation",
    "troubleshooting",
    "limitations",
    "roadmap",
)


class ManualLoadStatus(str, Enum):
    """Statuses returned while loading manual content."""

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    INVALID_MANUAL = "INVALID_MANUAL"
    READ_ERROR = "READ_ERROR"


@dataclass(frozen=True, slots=True)
class ManualSection:
    """One stable section in the internal manual."""

    id: str
    title: str
    path: Path
    summary: str
    tags: tuple[str, ...]
    order: int


@dataclass(frozen=True, slots=True)
class AtlasManualIndex:
    """Programmatic index for the internal Atlas manual."""

    sections: tuple[ManualSection, ...]
    version: str = MANUAL_VERSION


@dataclass(frozen=True, slots=True)
class ManualLoadResult:
    """Uniform result for manual loading."""

    status: ManualLoadStatus
    section: ManualSection | None
    content: str | None
    message: str


@dataclass(frozen=True, slots=True)
class ManualValidationIssue:
    """One deterministic manual validation issue."""

    code: str
    message: str
    section_id: str | None = None


@dataclass(frozen=True, slots=True)
class ManualValidationResult:
    """Result of validating the manual against project sources of truth."""

    valid: bool
    issues: tuple[ManualValidationIssue, ...]


class AtlasManualLoader:
    """Read the manual index and section files deterministically."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        index: AtlasManualIndex | None = None,
    ) -> None:
        self._root = root or MANUAL_ROOT
        self._index = index or default_manual_index(self._root)

    @property
    def index(self) -> AtlasManualIndex:
        """Return the configured manual index."""
        return self._index

    def list_sections(self) -> tuple[ManualSection, ...]:
        """Return sections in presentation order."""
        return tuple(sorted(self._index.sections, key=lambda section: section.order))

    def get_section(
        self,
        section_id: str,
    ) -> ManualLoadResult:
        """Load one section by stable id."""
        section = self._section_by_id(section_id)
        if section is None:
            return ManualLoadResult(
                status=ManualLoadStatus.NOT_FOUND,
                section=None,
                content=None,
                message=f"No existe la seccion del manual: {section_id}",
            )
        return self.load_content(section)

    def load_content(
        self,
        section: ManualSection,
    ) -> ManualLoadResult:
        """Load one section file without modifying it."""
        try:
            content = section.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return ManualLoadResult(
                status=ManualLoadStatus.NOT_FOUND,
                section=section,
                content=None,
                message=f"No existe el archivo de la seccion: {section.path}",
            )
        except OSError as error:
            return ManualLoadResult(
                status=ManualLoadStatus.READ_ERROR,
                section=section,
                content=None,
                message=str(error),
            )

        return ManualLoadResult(
            status=ManualLoadStatus.FOUND,
            section=section,
            content=content,
            message="Seccion cargada.",
        )

    def search(
        self,
        query: str,
    ) -> tuple[ManualSection, ...]:
        """Search by id, title, summary or tag using simple text matching."""
        normalized = _normalize(query)
        if not normalized:
            return ()

        matches = []
        for section in self.list_sections():
            haystack = " ".join(
                (
                    section.id,
                    section.title,
                    section.summary,
                    " ".join(section.tags),
                )
            )
            if normalized in _normalize(haystack):
                matches.append(section)
        return tuple(matches)

    def _section_by_id(
        self,
        section_id: str,
    ) -> ManualSection | None:
        normalized = section_id.strip().lower()
        for section in self._index.sections:
            if section.id == normalized:
                return section
        return None


class AtlasManualValidator:
    """Validate manual structure and selected facts against live registries."""

    def __init__(
        self,
        *,
        loader: AtlasManualLoader | None = None,
        tool_registry: ToolRegistry | None = None,
        schema_registry: ArgumentSchemaRegistry | None = None,
        tool_selector: ToolSelector | None = None,
    ) -> None:
        self._loader = loader or AtlasManualLoader()
        self._tool_registry = tool_registry or Bootstrap.build_tool_registry()
        self._schema_registry = schema_registry or Bootstrap.build_argument_schema_registry()
        self._tool_selector = tool_selector or Bootstrap.build_tool_selector(
            self._tool_registry
        )

    def validate(self) -> ManualValidationResult:
        """Validate the manual without executing registered tools."""
        issues: list[ManualValidationIssue] = []
        sections = self._loader.list_sections()
        issues.extend(_validate_index(sections))

        loaded: dict[str, str] = {}
        for section in sections:
            result = self._loader.load_content(section)
            if result.status is not ManualLoadStatus.FOUND or result.content is None:
                issues.append(
                    ManualValidationIssue(
                        "section_read_error",
                        result.message,
                        section.id,
                    )
                )
                continue
            loaded[section.id] = result.content
            issues.extend(_validate_section_content(section, result.content))

        if "tools" in loaded:
            issues.extend(
                self._validate_tools_documentation(
                    loaded["tools"],
                )
            )

        if "capabilities" in loaded:
            issues.extend(_validate_capability_states(loaded["capabilities"]))

        issues.extend(_validate_internal_links(loaded, sections))

        return ManualValidationResult(
            valid=not issues,
            issues=tuple(issues),
        )

    def _validate_tools_documentation(
        self,
        content: str,
    ) -> tuple[ManualValidationIssue, ...]:
        issues: list[ManualValidationIssue] = []
        documented = _parse_tools_table(content)
        registered = {descriptor.name: descriptor for descriptor in self._tool_registry.descriptors()}

        for tool_name in sorted(documented):
            if tool_name not in registered:
                issues.append(
                    ManualValidationIssue(
                        "unknown_tool",
                        f"Herramienta documentada no registrada: {tool_name}",
                        "tools",
                    )
                )

        for tool_name in sorted(registered):
            if tool_name not in documented:
                issues.append(
                    ManualValidationIssue(
                        "missing_tool",
                        f"Herramienta registrada sin documentar: {tool_name}",
                        "tools",
                    )
                )

        intent_to_tool = _intent_to_tool(self._tool_selector)
        schema_by_tool = _schema_summary_by_tool(
            self._schema_registry,
            intent_to_tool,
        )

        for tool_name, row in documented.items():
            descriptor = registered.get(tool_name)
            if descriptor is None:
                continue

            documented_confirmation = row.get("confirmacion", "").strip().upper()
            expected_confirmation = "SI" if descriptor.requires_confirmation else "NO"
            if documented_confirmation != expected_confirmation:
                issues.append(
                    ManualValidationIssue(
                        "confirmation_mismatch",
                        (
                            f"{tool_name}: confirmacion documentada "
                            f"{documented_confirmation}, real {expected_confirmation}"
                        ),
                        "tools",
                    )
                )

            expected_schema = schema_by_tool.get(tool_name, "sin schema conversacional registrado")
            documented_schema = row.get("argumentos", "").strip()
            if documented_schema != expected_schema:
                issues.append(
                    ManualValidationIssue(
                        "schema_mismatch",
                        (
                            f"{tool_name}: argumentos documentados "
                            f"{documented_schema!r}, reales {expected_schema!r}"
                        ),
                        "tools",
                    )
                )

        return tuple(issues)


def default_manual_index(
    root: Path = MANUAL_ROOT,
) -> AtlasManualIndex:
    """Return the static manual index used by loaders and CLIs."""
    entries = (
        ("overview", "Visión general", "Qué es Atlas y qué no es.", ("overview", "identity"), 10),
        ("architecture", "Arquitectura", "Componentes actuales y fronteras.", ("architecture", "bootstrap"), 20),
        ("capabilities", "Capacidades", "Estados reales de capacidades.", ("capabilities", "status"), 30),
        ("tools", "Herramientas", "Herramientas registradas y schemas.", ("tools", "registry", "schemas"), 40),
        ("execution_flow", "Flujo de ejecución", "Flujo de herramientas y cadenas.", ("execution", "chains"), 50),
        ("confirmations", "Confirmaciones", "Confirmaciones y modificaciones seguras.", ("confirmation", "safety"), 60),
        ("conversation", "Conversación", "Texto, aclaraciones y presentación.", ("conversation", "clarification"), 70),
        ("operation", "Operación", "Comandos de uso diario.", ("operation", "commands"), 80),
        ("troubleshooting", "Diagnóstico", "Problemas conocidos y comprobaciones.", ("troubleshooting", "diagnostics"), 90),
        ("limitations", "Limitaciones", "Límites actuales y capacidades no soportadas.", ("limitations", "safety"), 100),
        ("roadmap", "Roadmap", "Estado de fases y próximos pasos.", ("roadmap", "planning"), 110),
    )
    sections = tuple(
        ManualSection(
            id=section_id,
            title=title,
            path=root / f"{section_id}.md",
            summary=summary,
            tags=tags,
            order=order,
        )
        for section_id, title, summary, tags, order in entries
    )
    return AtlasManualIndex(sections=sections)


def _validate_index(
    sections: tuple[ManualSection, ...],
) -> tuple[ManualValidationIssue, ...]:
    issues: list[ManualValidationIssue] = []
    ids = [section.id for section in sections]
    orders = [section.order for section in sections]

    if len(ids) != len(set(ids)):
        issues.append(ManualValidationIssue("duplicate_id", "Hay ids duplicados."))

    if len(orders) != len(set(orders)) or orders != sorted(orders):
        issues.append(ManualValidationIssue("invalid_order", "El orden no es estable."))

    for section in sections:
        if not section.title.strip():
            issues.append(ManualValidationIssue("empty_title", "Titulo vacio.", section.id))
        if section.order <= 0:
            issues.append(ManualValidationIssue("invalid_order", "Orden no positivo.", section.id))
        if not section.path.exists():
            issues.append(
                ManualValidationIssue(
                    "missing_path",
                    f"No existe la ruta {section.path}",
                    section.id,
                )
            )

    for required_id in REQUIRED_SECTION_IDS:
        if required_id not in ids:
            issues.append(
                ManualValidationIssue(
                    "missing_required_section",
                    f"Falta la seccion obligatoria {required_id}.",
                    required_id,
                )
            )

    return tuple(issues)


def _validate_section_content(
    section: ManualSection,
    content: str,
) -> tuple[ManualValidationIssue, ...]:
    issues: list[ManualValidationIssue] = []
    if f"manual-id: {section.id}" not in content:
        issues.append(
            ManualValidationIssue(
                "missing_manual_id",
                f"La seccion no declara manual-id: {section.id}",
                section.id,
            )
        )
    if "Propósito:" not in content:
        issues.append(
            ManualValidationIssue(
                "missing_purpose",
                "La seccion no declara proposito.",
                section.id,
            )
        )
    if re.search(r"(sk-[A-Za-z0-9]|token\s*=|api[_-]?key\s*=)", content, re.IGNORECASE):
        issues.append(
            ManualValidationIssue(
                "possible_secret",
                "La seccion parece contener un secreto.",
                section.id,
            )
        )
    if "C:\\Users\\" in content:
        issues.append(
            ManualValidationIssue(
                "private_path",
                "La seccion contiene una ruta privada de usuario.",
                section.id,
            )
        )
    return tuple(issues)


def _validate_capability_states(
    content: str,
) -> tuple[ManualValidationIssue, ...]:
    issues: list[ManualValidationIssue] = []
    for match in re.finditer(r"\|\s*([A-Z_]+)\s*\|", content):
        state = match.group(1)
        if state in {"ESTADO", "CAPACIDAD"}:
            continue
        if state not in KNOWN_CAPABILITY_STATES:
            issues.append(
                ManualValidationIssue(
                    "unknown_capability_state",
                    f"Estado desconocido: {state}",
                    "capabilities",
                )
            )
    return tuple(issues)


def _validate_internal_links(
    loaded: Mapping[str, str],
    sections: tuple[ManualSection, ...],
) -> tuple[ManualValidationIssue, ...]:
    valid_files = {section.path.name for section in sections}
    issues: list[ManualValidationIssue] = []
    for section_id, content in loaded.items():
        for target in re.findall(r"\]\(([^)]+\.md)\)", content):
            if Path(target).name not in valid_files:
                issues.append(
                    ManualValidationIssue(
                        "broken_internal_link",
                        f"Enlace interno no resuelto: {target}",
                        section_id,
                    )
                )
    return tuple(issues)


def _parse_tools_table(
    content: str,
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    headers: list[str] | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if not cells:
            continue
        normalized_cells = [_normalize(cell).replace(" ", "_") for cell in cells]
        if "tool_name" in normalized_cells:
            headers = normalized_cells
            continue
        if set(cells[0]) <= {"-"}:
            continue
        if headers is None or len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        tool_name = row.get("tool_name", "")
        if tool_name:
            rows[tool_name] = row
    return rows


def _intent_to_tool(
    selector: ToolSelector,
) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for intent in selector.supported_intents():
        mappings[intent] = selector.select(ToolIntent(intent)).tool_name
    return mappings


def _schema_summary_by_tool(
    schemas: ArgumentSchemaRegistry,
    intent_to_tool: Mapping[str, str],
) -> dict[str, str]:
    summary: dict[str, str] = {}
    for intent, tool_name in intent_to_tool.items():
        schema = schemas.get(intent)
        required = [field.name for field in schema.fields if field.required]
        optional = [field.name for field in schema.fields if not field.required]
        parts = []
        if required:
            parts.append("req:" + ",".join(required))
        if optional:
            parts.append("opt:" + ",".join(optional))
        summary[tool_name] = "; ".join(parts) if parts else "sin argumentos"
    return summary


def _normalize(
    text: str,
) -> str:
    return " ".join(text.strip().lower().split())
