# Operación

manual-id: operation

Propósito: listar el comando oficial de arranque, requisitos y uso local.

## Requisitos mínimos

- Windows con PowerShell.
- Python 3.11, 3.12, 3.13 o 3.14.
- Dependencias críticas del modo texto: `ollama` y `numpy`.
- Dependencias opcionales para voz: `sounddevice`, `faster-whisper`,
  `pyttsx3` y `openwakeword`.
- Ejecutar desde la raíz `C:\AI\Atlas`.

Si se utiliza el entorno virtual incluido:

```powershell
Set-Location C:\AI\Atlas
.\.venv\Scripts\Activate.ps1
```

La planificación híbrida determinista, la ejecución estructurada y la
persistencia están activadas de forma predeterminada. Sus variables
`ATLAS_HYBRID_PLANNING_ENABLED`,
`ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED` y
`ATLAS_EXECUTION_PERSISTENCE_ENABLED` pueden establecerse en `false` como
opt-out de desarrollo.

## Comando oficial

```powershell
python -B main.py
```

Este es el único comando de arranque para el usuario. Los comandos de tests,
diagnóstico y desarrollo no son puntos de entrada alternativos.

## Escenario seguro verificado

```text
Tú: Lee README.md
```

La petición se clasifica como `SINGLE_TOOL`, selecciona `read_file`, genera y
valida un plan, selecciona estrategia, autoriza un único despacho y ejecuta la
lectura. La consola muestra un resultado breve; el informe operativo completo
permanece disponible para diagnóstico y persistencia.

Un objetivo de varios pasos puede solicitar una copia verificable:

```text
Tú: Lee README.md, guarda el contenido en .atlas/copia.txt y comprueba que se escribió correctamente.
Tú: confirmo
```

Atlas genera una cadena `read_file` → `write_file` → `read_file`. El contenido
del primer paso llega al segundo mediante una referencia estructurada, no por
interpolación de texto. El informe identifica la herramienta usada, la
referencia resuelta y el recurso producido. Además, la sección
`Verificación del objetivo` comprueba el número de pasos, las herramientas, la
existencia y lectura del recurso, la igualdad del contenido, la confirmación y
la ausencia de fallos críticos. Solo entonces muestra `VERIFIED`.

## Consultar herramientas

```text
Tú: ¿Qué herramientas tienes?
```

Atlas responde desde el `ToolRegistry` activo. La consulta no genera un plan ni
ejecuta herramientas. Las operaciones con escritura o control del escritorio
indican que requieren confirmación.

## Conversación diaria

Atlas acepta varias peticiones consecutivas dentro del mismo proceso. Después
de una lectura o de un error controlado puedes usar referencias como:

```text
Tú: Resume brevemente lo que acabas de leer.
Tú: ¿Qué archivo has leído?
Tú: ¿Cuál fue el error anterior?
```

Atlas conserva durante la sesión el contexto reciente y resúmenes acotados de
las ejecuciones. Si una referencia no tiene antecedente suficiente, solicita
aclaración en lugar de inventarlo. Un error esperado no cierra el bucle: puedes
enviar otra petición cuando reaparezca `Tú:`.

Cuando exista un plan pendiente, responde `confirmo` para ejecutarlo una sola
vez, `cancela`, `no`, `rechazar`, `olvidalo` o `no lo hagas` para descartarlo, o
`muestrame el plan` para revisarlo. Una consulta distinta no confirma la acción
pendiente. Al cerrar Atlas se descarta el contexto conversacional temporal; las
sesiones operativas persistidas y los informes conservan su ciclo independiente.

## Persistencia e informe

Las sesiones se guardan de forma predeterminada en:

```text
.atlas/execution_sessions/
```

La API interna `ExecutionSessionHistory.latest_execution()` devuelve la última
ejecución terminal. El informe se obtiene desde
`StructuredExecutionCoordinator.get_execution_report(session_id)` o desde
`AtlasOrchestrator.last_structured_execution_response.operational_report`
después de una petición estructurada.

Una sesión interrumpida conserva el plan, el contexto y los resultados
completados. Al reanudar, esos pasos no se repiten y las referencias posteriores
se resuelven desde el contexto restaurado.

## Confirmar una corrección

Una ejecución técnicamente completada puede devolver una segunda confirmación
si el objetivo no se verificó y existe una corrección determinista. Esa
confirmación autoriza solo el fragmento correctivo mostrado; la confirmación
del plan original ya consumida no se reutiliza.

El límite predeterminado es un ciclo, tres pasos, un recurso, una confirmación
nueva y una verificación final. Si no existe un valor esperado demostrado,
Atlas informa `INSUFFICIENT_EVIDENCE` y no escribe.

## Salir

Dentro del bucle de texto:

```text
salir
```

También se aceptan `exit` y `quit`.

## Preflight, logs y sesiones

Antes de arrancar, Atlas comprueba Python, archivos esenciales, directorios,
dependencias y configuración. Un `ERROR` bloquea el inicio y muestra una acción
recomendada. Un `WARNING` informa de una capacidad opcional no configurada, pero
no bloquea el modo texto.

- Log operativo: `logs/atlas.log`.
- Sesiones: `.atlas/execution_sessions/`.
- Estado reanudable: `.atlas/execution_state.json`.

El log rota al alcanzar 1 MB y conserva tres copias. Si no puede abrirse, Atlas
informa del modo degradado y continúa sin mostrar traceback.

## Codificación de PowerShell

La consola interactiva normal no necesita configuración adicional. Si una
redirección muestra caracteres españoles dañados, ejecuta antes:

```powershell
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
```

## Limitaciones actuales

- La voz permanece desactivada cuando faltan sus dependencias opcionales.
- Ollama debe estar iniciado para las respuestas que requieren modelo local.
- No existe instalador, ejecutable `.exe`, GUI ni autoarranque.
- Los comandos `salir`, `exit`, `quit`, Ctrl+C y EOF cierran el modo texto.

## Tests

```powershell
python -B -m pytest -q
```

## Estado Git

```powershell
git status
```

Los comandos Git deben ejecutarse fuera de Atlas, en PowerShell. Si Atlas está
esperando una confirmación, primero responde o cancela.

## Manual

```powershell
python -B -m tools.atlas_manual list
python -B -m tools.atlas_manual show overview
python -B -m tools.atlas_manual search confirmation
python -B -m tools.validate_atlas_manual
```
