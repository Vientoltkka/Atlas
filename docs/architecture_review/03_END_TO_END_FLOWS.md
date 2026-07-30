# Flujos end-to-end

## Flujo estructurado completo

```text
entrada de usuario
  -> main.py / Atlas.start()
  -> AtlasOrchestrator.start()
  -> RequestGateway.from_text()
  -> Router.classify_request() / RouteDecision
  -> Planner.generate_execution_plan()
  -> ExecutionPlanValidator.validate() + plan_signature
  -> HistoricalPlanAdjuster (opcional y conservador)
  -> ExecutionStrategySelector + Validator
  -> ExecutionAuthorizationGate
  -> ExecutionDispatcher
  -> StructuredExecutionCoordinator
  -> ExecutionPlanExecutor
  -> ToolExecutor
  -> ExecutionSupervisor
  -> GoalVerifier
  -> ObjectiveCorrection (solo si aplica, maximo un ciclo)
  -> FileExecutionSessionRepository / resumable store
  -> ExecutionReportGenerator
  -> AtlasOrchestrator._present_structured_execution()
  -> respuesta visible
```

El coordinador crea la sesion supervisada antes de ejecutar. El Dispatcher
consume la autorizacion antes de entregar el plan. El Executor usa
`ExecutionContext` y `ParameterResolver` para resolver cada paso.

## Consulta directa

```text
texto -> RequestGateway -> RouteDecision DIRECT_RESPONSE
      -> OperationalContextBuilder -> ChatAgent/PromptClient -> respuesta
```

Omisiones legitimas: Planner, Validator, Strategy, Authorization, Dispatcher,
Executor, Supervisor y GoalVerifier. No hay herramientas ni sesion ejecutiva.

## Catalogo y estado de capacidades

```text
texto -> detector determinista del AtlasOrchestrator -> ToolRegistry -> texto
```

Omisiones legitimas: RequestGateway puede no ser necesario, y no se ejecuta
ningun componente ejecutivo.

## Lectura segura

```text
"Lee README.md"
  -> SINGLE_TOOL
  -> plan de un paso read_file
  -> validacion/estrategia/autorizacion/despacho
  -> ToolExecutor
  -> Supervisor
  -> GoalVerifier INCONCLUSIVE si no hay criterios declarados
  -> informe y contenido visible
```

La finalizacion tecnica no se presenta como objetivo verificado.

## Multi-step con confirmacion

```text
read_file -> StepOutputReference -> write_file -> read_file
```

La escritura deja el plan en `WAITING_CONFIRMATION`. `confirmo` se vincula a la
firma y a los pasos protegidos; una entrada ambigua no ejecuta. Tras el
despacho, la confirmacion queda consumida.

## Cancelacion

`cancela`, `no` u otras frases cerradas eliminan el plan pendiente. No se
despacha ni se crea el recurso. La autorizacion no puede reutilizarse.

## Error y recuperacion conversacional

Un fallo de herramienta produce resultado tipado, sesion fallida e informe
sanitizado. El bucle de texto continua y una peticion posterior crea una nueva
ejecucion; la accion fallida no se repite automaticamente.

## Persistencia y reanudacion

```text
snapshot JSON -> RecoveryService -> policy conservadora
  -> restaurar ExecutionSession/ExecutionContext
  -> descartar pasos completados
  -> revalidar firma/schema
  -> nueva autorizacion si corresponde
  -> continuar pasos pendientes
```

Una sesion ambigua o con un paso en ejecucion requiere revision manual. Los
pasos completados y confirmaciones consumidas no se recrean.

## Verificacion y correccion

GoalVerifier evalua criterios y evidencia sin herramientas. Solo un
`NOT_VERIFIED` con valor esperado demostrado puede producir un fragmento
correctivo. El fragmento se valida, selecciona nueva estrategia, obtiene nueva
autorizacion y puede requerir otra confirmacion. No hay recursion.

## Ruta operacional tipada

`process_prompt_result()` usa `OperationalRouteExecutor` y handlers por
`RequestRoute`. Puede ejecutar DIRECT_RESPONSE, memoria, single tool, agent,
autonomous, resume, sistema o clarificacion. Es una ruta paralela de alto nivel
a `process_prompt()`, aunque reutiliza registros y nucleo ejecutivo.

## Ruta Atlas estructurada

`route_structured_input()` ejecuta normalizer -> classifier -> adapter ->
`AtlasRouter`. En v1.0 solo CAPABILITY y AGENT estan disponibles; CONVERSATION,
TOOL y WORKFLOW devuelven `ROUTE_UNAVAILABLE`.

## Fallback heredado

Si ninguna ruta anterior maneja la entrada, `_process_prompt_without_execution`
prueba escritorio, correccion de codigo, refactoring y finalmente Planner/Router
con agentes clasicos. Esta ruta conserva comportamiento previo y debe
considerarse explicitamente durante una futura consolidacion.
