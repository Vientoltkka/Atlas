# SPEC-0020 — Integración del Tool System

## Objetivo

Integrar completamente el sistema de herramientas existente con Atlas sin modificar el Core.

## Estado inicial

Implementado:

- BaseTool
- ToolRegistry
- ToolExecutor
- ToolContext
- ReadFileTool
- WriteFileTool
- ListDirectoryTool

Pendiente:

- Integración con CodingAgent
- Registro automático de herramientas
- Primer caso de uso funcional

## Criterios de aceptación

Atlas debe ser capaz de:

- Leer un archivo mediante ReadFileTool.
- Escribir un archivo mediante WriteFileTool.
- Listar un directorio mediante ListDirectoryTool.
- Mantener el Core sin modificaciones.

## Archivos afectados

tools/
agents/
bootstrap/
tests/

## Estado

EN DESARROLLO