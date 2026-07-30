# Contratos publicos

## Entrada y routing

| Contrato | Garantia principal |
|---|---|
| `AtlasRequest` | request tipada, inmutable y limitada |
| `RequestGateway` | normaliza texto/voz/sistema/reanudacion sin ejecutar |
| `RouteDecision` | seleccion tipada, trazable y sin efectos |
| `AtlasOrchestrator.process_prompt()` | respuesta visible de la ruta diaria |
| `AtlasOrchestrator.process_request()` | clasificar y ejecutar una request existente |
| `OperationalRouteExecutor.execute()` | ejecutar exactamente la decision recibida |
| `AtlasRoutingRequest/Result` | routing estructurado CAPABILITY/AGENT |

## Planning

| Contrato | Garantia principal |
|---|---|
| `ExecutionPlan` | goal, pasos, riesgos, criterios y confirmacion |
| `ExecutionStep` | tool, argumentos, dependencias, retry y bindings |
| `Planner` | produce plan o error tipado; no ejecuta |
| `PlanValidationResult` | validez, errores, warnings y firma |
| `plan_signature()` | SHA-256 determinista del plan validado |

## Argumentos y contexto

| Contrato | Garantia principal |
|---|---|
| `ExecutionArguments` | mapping defensivo, JSON-safe y sin NaN/Infinity |
| `StepOutputReference` | referencia tipada a un paso previo |
| `ExecutionVariableReference` | referencia tipada a variable de ejecucion |
| `ExecutionContext` | estados, resultados y variables de una ejecucion |
| `ExecutionContextSnapshot` | estado serializable y restaurable |
| `ParameterResolver` | resolucion acotada, defensiva y sin `eval` |

## Gobierno y ejecucion

| Contrato | Garantia principal |
|---|---|
| `ExecutionStrategySelectionResult` | estrategia validada sin alterar plan |
| `ExecutionAuthorizationResult` | autorizacion vinculada a firma y estrategia |
| `ExecutionConfirmationReference` | confirmacion ligada a plan/paso y TTL |
| `ExecutionDispatchResult` | permiso consumido y resultado de entrega |
| `ExecutionPlanExecutor.execute()` | ejecutar un plan ya validado |
| `PlanExecutionResult` | exito tecnico, pasos, metadata y verificacion |
| `ExecutionSession` | estado supervisado y transiciones |

## Persistencia y recuperacion

| Contrato | Garantia principal |
|---|---|
| `ExecutionSessionSnapshot` | snapshot versionado e inmutable |
| `ExecutionSessionRepository` | save/load/list/delete/exists |
| `FileExecutionSessionRepository` | JSON atomico por sesion |
| `RecoveryDecision` | resume, confirmacion, revision o terminal |
| `ResumableExecutionState` | checkpoint de executor |
| `JsonResumableExecutionStore` | checkpoint atomico y limitado |

## Resultado y verificacion

| Contrato | Garantia principal |
|---|---|
| `GoalVerificationResult` | estado independiente del exito tecnico |
| `GoalVerificationStatus` | VERIFIED, PARTIAL, NOT_VERIFIED, INCONCLUSIVE, USER_ACTION_REQUIRED |
| `ObjectiveCorrectionDecision` | correccion limitada o rechazo explicito |
| `OperationalExecutionReport` | informe serializable y sanitizado |

## Agentes, skills y capacidades

| Contrato | Estado v1.0 |
|---|---|
| `AgentDefinition`, `AgentRegistry` | declarativo, vacio por defecto |
| `AgentHandlerRegistry`, `AgentExecutor` | handlers explicitos, vacio por defecto |
| `MultiAgentCoordinator` | secuencial, experimental |
| `SkillSystem` | declarativo, experimental |
| `CapabilityExecutionRequest/Result` | ruta estructurada, experimental |
| `agents.registry.AgentRegistry` | registro clasico con chat/coding/project |

## Compatibilidad

Los snapshots y reportes incluyen defaults para payloads antiguos. Los tests
cubren lectura de sesiones legacy, cambios de schema, firmas, serializacion y
APIs de fases anteriores. No se garantiza compatibilidad futura fuera de los
contratos versionados.
