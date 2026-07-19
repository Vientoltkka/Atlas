# Visión general

manual-id: overview

Propósito: definir qué es Atlas en su estado actual y evitar prometer capacidades no implementadas.

## Qué es Atlas

Atlas es una aplicación local de asistente personal basada en Python. Su entrada principal actual es la consola con `python -B main.py`. Integra un orquestador, agentes, memoria en RAM, selección de modelos, herramientas registradas y flujos deterministas de ejecución de herramientas.

## Qué no es todavía

Atlas no es todavía un sistema autónomo general, no tiene RAG, no usa embeddings para consultar su documentación, no navega por Internet desde el flujo principal, no tiene memoria vectorial y no dispone de interfaz gráfica estable.

## Estado operativo

El sistema puede:

- responder por conversación directa mediante el flujo de agentes existente;
- detectar algunas peticiones de herramientas;
- pedir aclaraciones;
- pedir confirmación en acciones peligrosas;
- ejecutar herramientas únicas;
- ejecutar cadenas lineales deterministas;
- presentar resultados de herramientas en texto legible.

## Principio de seguridad

El usuario mantiene el control. Las acciones con escritura, tipeo o atajos confirmables se pausan antes de ejecutarse. Si hay una confirmación pendiente, la siguiente entrada se interpreta dentro de esa sesión y no como comando nuevo de PowerShell.
