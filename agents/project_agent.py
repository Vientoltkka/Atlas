"""Project analysis agent."""

from __future__ import annotations

import unicodedata

from agents.base_agent import BaseAgent

from models.prompt_client import PromptClient

from use_cases.find_project_file import FindProjectFileUseCase
from use_cases.read_file import ReadFileUseCase
from use_cases.read_project import ReadProjectUseCase
from use_cases.read_project_index import ReadProjectIndexUseCase
from use_cases.query_architecture_graph import (
    ArchitectureQueryResult,
    QueryArchitectureGraphUseCase,
)
from use_cases.resolve_project_dependencies import ResolveProjectDependenciesUseCase


class ProjectAgent(BaseAgent):
    """Agent specialized in project analysis."""

    SYSTEM_PROMPT = """
Eres Atlas Project Agent, un auditor profesional de codigo y arquitectura de software.

Analiza exclusivamente el codigo fuente recibido. Tu respuesta debe basarse solo en la informacion obtenida de los archivos leidos.

Debes:

- Explicar la arquitectura del proyecto.
- Detectar modulos y responsabilidades.
- Detectar dependencias internas y externas inferibles desde el codigo.
- Detectar codigo duplicado o patrones repetidos.
- Detectar malas practicas de diseno, mantenimiento, pruebas o estructura.
- Detectar posibles bugs, riesgos logicos o errores de integracion.
- Proponer mejoras tecnicas justificadas por el codigo analizado.
- No inventar archivos, dependencias, comportamiento ni decisiones arquitectonicas.
- Declarar explicitamente cualquier limitacion cuando el codigo leido no permita concluir algo.

Responde siempre en espanol, con lenguaje tecnico, claro y estructurado.
"""

    def __init__(
        self,
        prompt_client: PromptClient,
        read_project: ReadProjectUseCase,
        read_file: ReadFileUseCase | None = None,
        read_project_index: ReadProjectIndexUseCase | None = None,
        find_project_file: FindProjectFileUseCase | None = None,
        resolve_project_dependencies: ResolveProjectDependenciesUseCase | None = None,
        query_architecture_graph: QueryArchitectureGraphUseCase | None = None,
    ) -> None:

        self._client = prompt_client
        self._read_project = read_project
        self._read_file = read_file
        self._read_project_index = read_project_index or ReadProjectIndexUseCase()
        self._find_project_file = find_project_file or FindProjectFileUseCase(
            self._read_project_index
        )
        self._resolve_project_dependencies = (
            resolve_project_dependencies
            or ResolveProjectDependenciesUseCase()
        )
        self._query_architecture_graph = query_architecture_graph

    @property
    def name(self) -> str:
        return "project"

    @property
    def description(self) -> str:
        return "Project analysis."

    def run(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> str:
        user_query = self._last_user_query(messages)
        architecture_query = self._answer_architecture_query(
            query=user_query,
        )

        if architecture_query is not None:
            return architecture_query

        file_match = self._find_file_before_full_analysis(
            query=user_query,
            model=model,
        )

        if file_match is not None:
            return file_match

        project = self._read_project.execute(".")

        summary: list[str] = []

        for file in project:

            summary.append(f"Archivo: {file['path']}")
            summary.append("")
            summary.append("Código:")
            summary.append("")
            summary.append(file["content"])
            summary.append("")
            summary.append("-" * 80)
            summary.append("")

        conversation = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "\n".join(summary),
            },
        ]

        return self._client.ask(
            model=model,
            messages=conversation,
        )

    def _last_user_query(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Return the most recent user message content."""
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")

        return ""

    def _answer_architecture_query(
        self,
        query: str,
    ) -> str | None:
        """Answer supported architecture graph queries deterministically."""
        if self._query_architecture_graph is None:
            return None

        intent = self._architecture_intent(query)

        if intent is None:
            return None

        operation, target = intent

        if operation == "dependents":
            result = self._query_architecture_graph.dependents_of(target)
            return self._format_dependents_result(result)

        if operation == "dependencies":
            result = self._query_architecture_graph.dependencies_of(target)
            return self._format_dependencies_result(result)

        if operation == "imported_classes":
            result = self._query_architecture_graph.imported_classes_of(target)
            return self._format_imported_classes_result(result)

        if operation == "impact":
            result = self._query_architecture_graph.impact_of(target)
            return self._format_impact_result(result)

        return None

    def _architecture_intent(
        self,
        query: str,
    ) -> tuple[str, str] | None:
        """Detect supported architecture graph intents."""
        normalized_query = self._normalize_text(query)
        target = self._extract_architecture_target(query)

        if target is None:
            return None

        if "quien usa" in normalized_query:
            return ("dependents", target)

        if "dependen de" in normalized_query or "depende de" in normalized_query:
            return ("dependents", target)

        if "clases importa" in normalized_query:
            return ("imported_classes", target)

        if "afectados" in normalized_query or "afectadas" in normalized_query:
            return ("impact", target)

        if "dependencias de" in normalized_query or "que usa" in normalized_query:
            return ("dependencies", target)

        return None

    def _extract_architecture_target(
        self,
        query: str,
    ) -> str | None:
        """Extract the likely architecture target from a user query."""
        tokens = [
            token.strip(".,:;()[]{}¿?¡!\"'`")
            for token in query.split()
        ]

        candidates = [
            token
            for token in tokens
            if token
            and (
                ".py" in token
                or "." in token
                or token[:1].isupper()
                or "_" in token
            )
        ]

        if not candidates:
            return None

        return candidates[-1]

    def _format_dependents_result(
        self,
        result: ArchitectureQueryResult,
    ) -> str:
        """Format a dependents query result."""
        unresolved = self._format_unresolved_result(result)

        if unresolved is not None:
            return unresolved

        return "\n".join(
            [
                "Objetivo:",
                str(result.target),
                "",
                "Dependientes directos:",
                *self._format_bullets(result.direct_dependents),
                "",
                "Dependientes indirectos:",
                *self._format_bullets(result.indirect_dependents),
            ]
        )

    def _format_dependencies_result(
        self,
        result: ArchitectureQueryResult,
    ) -> str:
        """Format a dependencies query result."""
        unresolved = self._format_unresolved_result(result)

        if unresolved is not None:
            return unresolved

        return "\n".join(
            [
                "Objetivo:",
                str(result.target),
                "",
                "Dependencias internas:",
                *self._format_bullets(result.dependencies),
            ]
        )

    def _format_imported_classes_result(
        self,
        result: ArchitectureQueryResult,
    ) -> str:
        """Format an imported classes query result."""
        unresolved = self._format_unresolved_result(result)

        if unresolved is not None:
            return unresolved

        return "\n".join(
            [
                "Objetivo:",
                str(result.target),
                "",
                "Clases importadas:",
                *self._format_bullets(result.imported_classes),
            ]
        )

    def _format_impact_result(
        self,
        result: ArchitectureQueryResult,
    ) -> str:
        """Format an impact query result."""
        unresolved = self._format_unresolved_result(result)

        if unresolved is not None:
            return unresolved

        return "\n".join(
            [
                "Objetivo:",
                str(result.target),
                "",
                "Archivos potencialmente afectados:",
                *self._format_bullets(result.affected_files),
            ]
        )

    def _format_unresolved_result(
        self,
        result: ArchitectureQueryResult,
    ) -> str | None:
        """Format missing or ambiguous architecture targets."""
        if result.target is not None:
            return None

        if result.matches:
            return "\n".join(
                [
                    "Hay varias coincidencias posibles.",
                    "Concreta el objetivo:",
                    *self._format_bullets(result.matches),
                ]
            )

        return "No se ha encontrado el objetivo en el grafo arquitectonico."

    def _format_bullets(
        self,
        values: list[str],
    ) -> list[str]:
        """Format values as bullets or an explicit empty value."""
        if not values:
            return ["- ninguno"]

        return [f"- {value}" for value in values]

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """Normalize accents and punctuation for intent detection."""
        normalized = unicodedata.normalize("NFKD", text.lower())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )

        return without_accents.translate(
            str.maketrans("¿?¡!", "    ")
        )

    def _find_file_before_full_analysis(
        self,
        query: str,
        model: str,
    ) -> str | None:
        """Locate a concrete file, class, or function before project analysis."""
        if not self._looks_like_file_lookup(query):
            return None

        index = self._read_project_index.execute(".")
        result = self._find_project_file.execute(
            query,
            index=index,
        )

        if isinstance(result, str):
            return self._analyze_file(
                model=model,
                path=result,
                index=index,
            )

        if isinstance(result, list):
            return "Archivos encontrados:\n" + "\n".join(result)

        return None

    def _analyze_file(
        self,
        model: str,
        path: str,
        index: list[dict[str, object]],
    ) -> str:
        """Analyze one concrete file and its direct internal dependencies."""
        if self._read_file is None:
            return f"Archivo encontrado:\n{path}"

        paths = self._resolve_project_dependencies.execute(
            main_path=path,
            index=index,
        )
        self._build_file_analysis_prompt(paths)
        analysis = self._build_local_file_analysis(
            paths=paths,
            index=index,
        )

        return f"Archivo encontrado:\n{path}\n\n{analysis}"

    def _build_local_file_analysis(
        self,
        paths: list[str],
        index: list[dict[str, object]],
    ) -> str:
        """Build a deterministic analysis from indexed file metadata."""
        main_path = paths[0]
        dependency_paths = paths[1:]
        main_item = self._find_index_item(main_path, index)

        sections = [
            "Análisis del archivo principal:",
            f"- Archivo principal: {main_path}",
            f"- Dependencias internas directas: {len(dependency_paths)}",
        ]

        if dependency_paths:
            sections.append(
                "- Rutas de dependencias: "
                + ", ".join(dependency_paths)
            )
        else:
            sections.append("- Rutas de dependencias: ninguna")

        if main_item is not None:
            sections.extend(
                [
                    "- Clases detectadas: "
                    + self._format_names(main_item.get("classes", [])),
                    "- Funciones detectadas: "
                    + self._format_names(main_item.get("functions", [])),
                    "- Imports detectados: "
                    + self._format_names(main_item.get("imports", [])),
                ]
            )

        sections.extend(
            [
                "",
                "Contexto usado:",
                "- Se ha leído el archivo principal.",
                "- Se han leído únicamente sus dependencias internas directas.",
                "- No se ha cargado el proyecto completo.",
            ]
        )

        return "\n".join(sections)

    def _find_index_item(
        self,
        path: str,
        index: list[dict[str, object]],
    ) -> dict[str, object] | None:
        """Find one indexed item by path."""
        for item in index:
            if str(item["path"]).replace("\\", "/") == path:
                return item

        return None

    def _format_names(
        self,
        values: object,
    ) -> str:
        """Format indexed names for a readable response."""
        if not isinstance(values, list) or not values:
            return "ninguno"

        return ", ".join(str(value) for value in values)

    def _build_file_analysis_prompt(
        self,
        paths: list[str],
    ) -> str:
        """Build the prompt for one file plus direct dependencies."""
        if self._read_file is None:
            return ""

        main_path = paths[0]
        sections = [
            "Analiza el archivo principal usando el codigo recibido.",
            "No analices archivos que no aparezcan en este prompt.",
            "",
            "Archivo principal:",
            main_path,
            "",
            "Codigo del archivo principal:",
            self._read_file.execute(main_path),
        ]

        dependency_paths = paths[1:]

        if dependency_paths:
            sections.extend(
                [
                    "",
                    "Dependencias internas directas:",
                ]
            )

            for dependency_path in dependency_paths:
                sections.extend(
                    [
                        "",
                        "Dependencia:",
                        dependency_path,
                        "",
                        "Codigo de la dependencia:",
                        self._read_file.execute(dependency_path),
                    ]
                )

        return "\n".join(sections)

    def _looks_like_file_lookup(
        self,
        query: str,
    ) -> bool:
        """Detect concrete lookup requests without blocking normal analysis."""
        normalized_query = query.strip()

        if not normalized_query:
            return False

        lowered_query = normalized_query.lower()

        if ".py" in lowered_query or "/" in normalized_query or "\\" in normalized_query:
            return True

        tokens = [
            token.strip(".,:;()[]{}¿?¡!\"'`")
            for token in normalized_query.split()
            if token.strip(".,:;()[]{}¿?¡!\"'`")
        ]

        if len(tokens) == 1:
            return True

        lookup_words = {
            "archivo",
            "clase",
            "funcion",
            "función",
            "ruta",
            "path",
            "localiza",
            "encuentra",
            "donde",
            "dónde",
        }

        return any(token.lower() in lookup_words for token in tokens)
