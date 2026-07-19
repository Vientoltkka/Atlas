# Manual interno de Atlas

manual-id: index

Propósito: servir como punto de entrada humano al manual interno de Atlas 5.5A.

## Objetivo

Este manual es la fuente interna de verdad consultable para describir qué es Atlas, qué hace hoy, qué no hace todavía y cómo operarlo sin confundir capacidades implementadas con planes.

## Fuentes de verdad

- Código de `bootstrap.bootstrap.Bootstrap`.
- `ToolRegistry`, `ToolIntentRegistry` y `ArgumentSchemaRegistry`.
- Tests existentes bajo `tests/`.
- Documentación existente en `README.md`, `docs/`, `ROADMAP.md`, `TASKS.md` y `specs/`.
- Comportamiento validado por comandos locales.

## Secciones

1. [Visión general](overview.md)
2. [Arquitectura](architecture.md)
3. [Capacidades](capabilities.md)
4. [Herramientas](tools.md)
5. [Flujo de ejecución](execution_flow.md)
6. [Confirmaciones](confirmations.md)
7. [Conversación](conversation.md)
8. [Operación](operation.md)
9. [Diagnóstico](troubleshooting.md)
10. [Limitaciones](limitations.md)
11. [Roadmap](roadmap.md)

## Política de actualización

- Si cambia una herramienta registrada, actualizar `tools.md` y ejecutar `python -B -m tools.validate_atlas_manual`.
- Si cambia una fase de ejecución, actualizar `execution_flow.md`, `confirmations.md`, `conversation.md` o `roadmap.md`.
- Si una capacidad solo está planificada, marcarla como `PLANNED`; si existe pero no cubre todos los casos, marcarla como `PARTIAL`.
- No añadir secretos, tokens, claves, credenciales ni rutas privadas personales.
- No documentar como terminado nada que no esté implementado o probado.

## Validación automática

El validador comprueba ids únicos, rutas existentes, secciones obligatorias, enlaces internos, estados de capacidades, herramientas registradas, argumentos de schemas conversacionales y coincidencia de `requires_confirmation`.

## Contenido manual

Las explicaciones de arquitectura, diagnóstico, operación y limitaciones son manuales. La lista de herramientas es manual pero se valida contra el registry para reducir obsolescencia.
