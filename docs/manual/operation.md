# Operación

manual-id: operation

Propósito: listar el comando oficial de arranque, requisitos y uso local.

## Requisitos mínimos

- Windows con PowerShell.
- Python y las dependencias de `requirements.txt`.
- Ejecutar desde la raíz del repositorio.

La planificación híbrida determinista, la ejecución estructurada y la
persistencia están activadas de forma predeterminada. Sus variables
`ATLAS_HYBRID_PLANNING_ENABLED`,
`ATLAS_STRUCTURED_PLAN_EXECUTION_ENABLED` y
`ATLAS_EXECUTION_PERSISTENCE_ENABLED` pueden establecerse en `false` como
opt-out de desarrollo.

## Comando oficial

```powershell
python main.py
```

`python -B main.py` es una variante de desarrollo que evita generar bytecode,
pero no es un segundo punto de entrada.

## Escenario seguro verificado

```text
Tú: Lee README.md
```

La petición se clasifica como `SINGLE_TOOL`, selecciona `read_file`, genera y
valida un plan, selecciona estrategia, autoriza un único despacho, ejecuta la
lectura y presenta el informe operativo con el resultado real.

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
