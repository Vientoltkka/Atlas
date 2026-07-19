# Flujo de ejecución

manual-id: execution_flow

Propósito: describir cómo Atlas decide, propone, confirma, ejecuta y presenta herramientas.

## Modos

- `DIRECT_RESPONSE`: no se usa herramienta registrada y el turno vuelve al flujo conversacional.
- `SINGLE_TOOL`: una intención registrada puede resolver la petición.
- `TOOL_CHAIN`: varias intenciones registradas se ejecutan en orden lineal.

## Herramienta única

1. `ExecutionDecisionEngine` detecta candidato.
2. `ToolProposalBuilder` extrae argumentos.
3. `ArgumentValidator` valida schema.
4. `SingleToolRunner` comprueba si requiere confirmación.
5. Si no requiere confirmación, ejecuta mediante `ToolExecutor`.
6. `ExecutionResultPresenter` presenta el resultado.

## Cadena lineal

1. `ToolChainProposalBuilder` crea pasos ordenados.
2. Las referencias usan `${steps.<id>.output}` o campos soportados.
3. `ToolChainRunner` ejecuta desde el primer paso.
4. Si un paso requiere confirmación, la cadena queda pausada.
5. Al confirmar, continúa desde el paso pendiente sin repetir pasos previos.

## Estados de coordinación

`DIRECT_RESPONSE_REQUIRED`, `INFORMATION_REQUIRED`, `AMBIGUOUS_REQUEST`, `UNSUPPORTED`, `VALIDATION_FAILED`, `CONFIRMATION_REQUIRED`, `EXECUTED`, `CANCELLED`, `FAILED`.

## Límites

No hay loops, ramas, paralelismo, retries, rollback ni ejecución autónoma. Las cadenas son deterministas y finitas.
