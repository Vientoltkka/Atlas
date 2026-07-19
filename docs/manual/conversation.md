# Conversación

manual-id: conversation

Propósito: explicar el comportamiento conversacional de texto, aclaraciones y presentación de resultados.

## Flujo de texto

`ExecutionConversationController` se ejecuta antes del flujo conversacional clásico. Si detecta una acción soportada, gestiona aclaraciones, confirmaciones y presentación. Si no, devuelve `DIRECT_RESPONSE_REQUIRED` y el orquestador usa agentes.

## Aclaraciones multi-turno

Cuando faltan campos, Atlas pregunta de forma natural. Ejemplos:

- `Lee este archivo` -> pregunta qué archivo leer.
- `Escribe algo en un archivo` -> pregunta contenido y destino.
- `Lee este archivo y guárdalo en otro` -> pregunta origen y destino.

Las respuestas vacías o irrelevantes no ejecutan herramientas.

## Presentación

`ExecutionResultPresenter` evita mostrar dataclasses, ids internos o detalles técnicos en modo normal. Para resultados largos muestra una vista parcial sin modificar el resultado interno.

## Diferencias con voz

La conversación de texto es la ruta más estable para operaciones de herramientas. Voz y wake word existen parcialmente, pero pueden fallar por captura, STT, TTS o dispositivo.
