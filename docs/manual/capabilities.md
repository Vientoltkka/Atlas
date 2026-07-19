# Capacidades

manual-id: capabilities

Propósito: clasificar capacidades reales de Atlas por estado operativo.

| Capacidad | Estado | Evidencia | Límite |
|---|---|---|---|
| conversación directa | IMPLEMENTED | `AtlasOrchestrator` y agentes | depende del modelo configurado |
| lectura de archivos | IMPLEMENTED | `file.read` -> `read_file` | texto UTF-8 |
| escritura de archivos | IMPLEMENTED | `file.write` -> `write_file` | requiere confirmación |
| listado de directorios | IMPLEMENTED | `directory.list` -> `list_directory` | lista nombres, no metadatos |
| árbol de proyecto | IMPLEMENTED | `project.tree` -> `project_tree` | devuelve archivos Python |
| escritorio | PARTIAL | herramientas `desktop.*` registradas | no todas tienen intent conversacional |
| ventanas | PARTIAL | listar ventanas y activar/mover/redimensionar herramientas | cobertura conversacional limitada |
| portapapeles | PARTIAL | herramientas registradas | no todas expuestas al selector conversacional |
| procesos | PARTIAL | listar, consultar, cerrar y terminar procesos | acciones destructivas deben tratarse con cautela |
| voz | PARTIAL | use cases de STT, TTS, wake word y assistant | reconocimiento y wake word pueden ser irregulares |
| herramienta única | IMPLEMENTED | `SingleToolRunner` | solo intents registrados |
| cadenas lineales | IMPLEMENTED | `ToolChainRunner` | sin ramas, loops ni paralelismo |
| aclaraciones multi-turno | IMPLEMENTED | `PendingClarification` | campos simples y cadenas soportadas |
| confirmaciones | IMPLEMENTED | `PendingToolConfirmation` | una operación pendiente por sesión |
| modificación de confirmaciones | IMPLEMENTED | `PendingConfirmationResolver` | no cambia pasos ya ejecutados |
| presentación de resultados | IMPLEMENTED | `ExecutionResultPresenter` | no resume semánticamente sin LLM |
| RAG del manual | PLANNED | fase posterior | no existe en 5.5A |
| búsqueda semántica | UNSUPPORTED | sin embeddings | solo búsqueda determinista del manual |
| navegación web | UNSUPPORTED | no hay tool web en registry actual | plan aspiracional en docs antiguos |
| memoria vectorial | UNSUPPORTED | no existe vector store | roadmap antiguo lo lista como futuro |
