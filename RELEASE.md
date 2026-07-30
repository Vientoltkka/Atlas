# Atlas Base v1.0

Fecha de congelacion: 2026-07-30

Estado: CONGELADA

Fase final: FASE 15.8

## Estado del proyecto

Atlas Base v1.0 es la linea base operativa validada para Windows. La evolucion
hacia V2 debe partir de esta version sin reinterpretar capacidades planificadas
como capacidades disponibles.

## Capacidades disponibles

- Arranque oficial con preflight y banner desde `python -B main.py`.
- Conversacion por texto con contexto reciente limitado.
- Consulta determinista del catalogo activo de herramientas.
- Lectura y escritura de texto UTF-8, listado de directorios y arbol Python.
- Planificacion estructurada determinista, validacion y argumentos tipados.
- Ejecucion secuencial de herramientas y planes multi-step.
- Estrategia, autorizacion, confirmacion explicita y cancelacion.
- Reintentos limitados, incluido el contrato `NO_RETRY`.
- Supervisor, persistencia local, recarga y reanudacion sin repetir pasos.
- Informes operativos y verificacion separada del exito tecnico.
- Correccion determinista no recursiva y limitada a un ciclo.
- Agentes, herramientas de escritorio y rutas de voz ya registradas en la base.

Las herramientas de escritorio y la voz son capacidades parciales: su
disponibilidad depende del entorno y no todas las herramientas tienen un intent
conversacional.

## Capacidades no disponibles

- Interfaz grafica, instalador, ejecutable autonomo o autoarranque.
- Integracion con WhatsApp.
- Navegacion web desde el flujo principal.
- RAG, embeddings o memoria vectorial.
- Operacion distribuida, multiusuario o sincronizacion entre dispositivos.
- Rollback general de efectos externos.
- Correccion semantica general o recursiva.

## Requisitos minimos

- Windows con PowerShell.
- Python 3.11, 3.12, 3.13 o 3.14.
- Ejecucion desde la raiz del repositorio.
- Ollama iniciado para respuestas que utilicen el modelo local.

## Dependencias

Dependencias declaradas: `ollama`, `openai`, `python-dotenv`, `pydantic`,
`rich`, `typer`, `fastapi`, `uvicorn`, `numpy`, `sounddevice`,
`faster-whisper`, `openwakeword==0.6.0` y `pyttsx3`.

El modo texto requiere principalmente `ollama` y `numpy`. Las dependencias de
audio y wake word son opcionales para el uso diario por texto.

## Limitaciones conocidas

- La latencia de conversacion depende del modelo y del equipo local.
- Un plan sin evidencia determinista suficiente termina como `INCONCLUSIVE`.
- Las confirmaciones pendientes deben resolverse antes de iniciar otro plan.
- La idempotencia del dispatcher es local, no distribuida.
- No existe rollback general para efectos externos.
- `read_file` y `write_file` conservan texto, no identidad binaria.
- La voz solo tiene cobertura smoke y puede quedar desactivada.

## Arranque oficial

```powershell
Set-Location C:\AI\Atlas
python -B main.py
```

## Validacion

La comprobacion rapida de la version esta en `CHECKLIST_FINAL.md`. La
congelacion oficial exige ademas `py_compile`, `compileall`, la suite completa
de `pytest` y `git diff --check`.
