# Limitaciones

manual-id: limitations

Propósito: registrar límites actuales para evitar sobreprometer capacidades.

## Límites de 5.5A

- El manual no está conectado al bucle conversacional principal.
- No hay RAG, embeddings ni búsqueda semántica.
- No hay LLM leyendo el manual.
- La búsqueda del CLI es determinista por título, resumen, id o tag.
- La validación no ejecuta herramientas.

## Límites de ejecución

- Solo se soportan intents registrados.
- Las cadenas son lineales.
- No hay rollback automático.
- No hay paralelismo.
- Las confirmaciones pendientes bloquean el siguiente turno.

## Límites de herramientas

- Muchas herramientas de escritorio están registradas pero no tienen intent conversacional ni schema.
- No hay herramienta web registrada.
- No hay herramienta de hora/fecha registrada.
- No hay herramienta de eliminación de archivos registrada.

## Límites de documentación heredada

`ROADMAP.md`, `TASKS.md` y `docs/ARCHITECTURE.md` contienen objetivos o aspiraciones que no siempre reflejan el estado real. Este manual debe actuar como capa de estado actual y enlazar esos documentos sin convertirlos automáticamente en verdad operativa.
