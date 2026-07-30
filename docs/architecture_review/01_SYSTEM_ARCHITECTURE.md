# Arquitectura del sistema

## Forma general

Atlas es una aplicacion Python local compuesta manualmente por
`bootstrap/bootstrap.py`. No usa un contenedor de inyeccion de dependencias.
Los contratos principales son dataclasses tipadas, enums y servicios
inyectados.

## Capas reales

| Capa | Componentes principales | Responsabilidad |
|---|---|---|
| Arranque | `main.py`, `core/startup.py`, `core/atlas.py` | preflight, logging, modos y ciclo de vida |
| Composicion | `bootstrap/bootstrap.py` | construir registros, servicios y orquestadores |
| Entrada | `core/request_gateway.py` | normalizar texto, voz, sistema y reanudacion |
| Routing diario | `core/router.py`, `core/operational_request_router.py` | clasificar sin ejecutar |
| Orquestacion CLI | `core/orchestrator.py` | ordenar rutas y presentar resultados |
| Planning | `core/planner.py`, planners hibrido y determinista | producir `ExecutionPlan` |
| Validacion | `core/execution_plan_validator.py` | validar y firmar planes |
| Gobierno | estrategia, autorizacion, dispatcher | seleccionar controles y permitir un despacho |
| Ejecucion | `core/execution_plan_executor.py` | ejecutar pasos mediante `ToolExecutor` |
| Supervision | `core/execution_supervisor.py` | estado de sesion, pasos, eventos y resumen |
| Verificacion | `core/goal_verifier.py` | separar exito tecnico de objetivo satisfecho |
| Correccion | `core/objective_correction.py` | un fragmento determinista y acotado |
| Persistencia | repositorio de sesiones y resumable store | snapshots, recuperacion y reanudacion |
| Presentacion | `core/execution_report.py`, presenter del orquestador | informes y respuesta segura |

## Composicion efectiva

`Bootstrap.build()` crea una instancia compartida de:

- `ToolRegistry`, `ToolExecutor` y schemas;
- `Planner`, `ExecutionPlanValidator` y `ExecutionPlanExecutor`;
- `ExecutionSupervisor` y repositorio de sesiones;
- `ExecutionStrategySelector`;
- `ExecutionAuthorizationGate` y `ExecutionDispatcher`;
- `ExecutionHistoryAdvisor` y `HistoricalPlanAdjuster`.

Esas instancias se inyectan en `StructuredExecutionCoordinator`. El
`AutonomousExecutionOrchestrator` recibe el mismo Planner, Validator, Executor,
Supervisor, Strategy, Authorization, Dispatcher e historial. Esto es
reutilizacion del nucleo, no un segundo motor independiente.

## Fachadas superiores

| Fachada | Ruta principal | Estado |
|---|---|---|
| `AtlasOrchestrator` | CLI y API conversacional | estable |
| `StructuredExecutionCoordinator` | planes estructurados diarios | estable |
| `OperationalRouteExecutor` | `process_prompt_result()` y rutas tipadas | estable, ruta alternativa |
| `AutonomousExecutionOrchestrator` | handler operacional autonomo | experimental |
| `CapabilityOrchestrator` | requests de capacidades | experimental |
| `AtlasRouter` | requests estructuradas CAPABILITY/AGENT | experimental |
| `AgentExecutor` | handlers declarativos | experimental; vacio por defecto |
| `SkillExecutor` | skills declarativas | experimental |

## Limites de responsabilidad

- Planner produce planes; no ejecuta.
- Validator valida estructura, referencias, politicas y firma; no ejecuta.
- Strategy selecciona configuracion; no modifica el plan.
- Authorization vincula plan, firma, estrategia y confirmaciones.
- Dispatcher consume una autorizacion y entrega una vez dentro del proceso.
- Executor resuelve argumentos y ejecuta herramientas; no decide la ruta.
- Supervisor registra estado; no ejecuta herramientas.
- GoalVerifier evalua evidencia; no ejecuta.
- Orchestrator ordena y presenta; no implementa herramientas.

## Riesgo de solapamiento

La separacion interna es clara, pero la capa superior tiene mas de una entrada:

- `start()` y `process_prompt()` priorizan catalogo, capacidades, conversacion
  directa y coordinador estructurado.
- `process_prompt_result()` clasifica y delega a `OperationalRouteExecutor`.
- `_process_prompt_without_execution()` conserva escritorio, correccion de
  codigo, refactoring y agentes clasicos.
- `route_structured_input()` usa normalizador, clasificador, adapter y
  `AtlasRouter`.

V2 debe decidir cuales de estas entradas son publicas y cuales son
compatibilidad, sin eliminar controles durante la consolidacion.
