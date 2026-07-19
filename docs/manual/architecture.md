# Arquitectura

manual-id: architecture

Propósito: explicar los componentes actuales de Atlas y sus fronteras sin volcar código.

## Composition root

`bootstrap.bootstrap.Bootstrap` construye los componentes principales: router, planner, memoria, agentes, modelos, herramientas, use cases de voz, escritorio y ejecución conversacional.

## Flujo principal de texto

Usuario
-> `main.py`
-> `core.atlas.Atlas`
-> `AtlasOrchestrator`
-> `ExecutionConversationController`
-> `ExecutionCoordinator`
-> `ExecutionDecisionEngine`
-> `ToolProposalBuilder` o `ToolChainProposalBuilder`
-> `SingleToolRunner` o `ToolChainRunner`
-> `ExecutionResultPresenter`

Si `ExecutionDecisionEngine` devuelve `DIRECT_RESPONSE`, el flujo vuelve al comportamiento conversacional previo del orquestador, router, memoria y agentes.

## Responsabilidades

- `AtlasOrchestrator`: coordina el turno de usuario y decide si usa ejecución conversacional o agentes.
- `ExecutionConversationController`: conserva estado de aclaración y confirmación de una sesión textual.
- `ExecutionCoordinator`: integra decisión, propuesta, validación y runners.
- `ExecutionDecisionEngine`: clasifica el modo sin ejecutar herramientas.
- `ToolProposalBuilder`: extrae argumentos de una herramienta única sin ejecutar.
- `ToolChainProposalBuilder`: construye cadenas lineales con referencias entre pasos.
- `SingleToolRunner`: selecciona, valida, confirma si procede y ejecuta una herramienta.
- `ToolChainRunner`: ejecuta pasos lineales, pausa en confirmaciones y no repite pasos ya completados.
- `ExecutionResultPresenter`: convierte resultados estructurados en texto de consola.

## Fronteras

Los builders no ejecutan. El presenter no ejecuta ni modifica estado. El coordinator no llama a `ToolExecutor` directamente. Los runners son los únicos componentes de esta línea que llegan a ejecutar herramientas mediante el executor.

## Documentación existente

`docs/ARCHITECTURE.md` contiene visión aspiracional y puede mencionar capacidades no implementadas. Este manual distingue explícitamente entre `IMPLEMENTED`, `PARTIAL`, `PLANNED` y `UNSUPPORTED`.
