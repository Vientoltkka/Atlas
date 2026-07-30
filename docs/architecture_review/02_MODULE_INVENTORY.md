# Inventario de modulos

## Criterio

Se detectaron 207 modulos Python en los subsistemas de produccion. Este
inventario enumera los modulos que definen arquitectura, contratos o rutas de
ejecucion. Las hojas auxiliares se agrupan al final.

Estados usados:

- `estable`: forma parte del flujo base validado.
- `experimental`: existe y tiene pruebas, pero no domina la CLI diaria.
- `legado`: se conserva por compatibilidad.
- `pendiente`: infraestructura sin activacion operativa completa.

## Arranque, composicion y routing

| Archivo | Responsabilidad | Dependencias principales | Contratos publicos | Estado | Pruebas |
|---|---|---|---|---|---|
| `main.py` | entrada Windows y ciclo de vida | startup, Atlas | `main()` | estable | `test_windows_startup.py`, `test_operational_end_to_end.py` |
| `core/atlas.py` | fachada de aplicacion | Bootstrap | `Atlas` | estable | startup y E2E |
| `bootstrap/bootstrap.py` | composition root | todos los subsistemas | `Bootstrap` | estable | `test_bootstrap.py`, E2E |
| `core/startup.py` | preflight, banner y logging | stdlib | `WindowsStartupPreflight`, renderers | estable | `test_windows_startup.py` |
| `core/request_gateway.py` | requests tipadas y limites | stdlib | `AtlasRequest`, `RequestGateway` | estable | `test_request_gateway.py` |
| `core/operational_request_router.py` | clasificacion determinista | registries | `RouteDecision`, `OperationalRequestRouter` | estable | `test_operational_request_router.py` |
| `core/router.py` | compatibilidad y delegacion operacional | Planner, router operacional | `Router` | legado/estable | `test_router.py` |
| `core/orchestrator.py` | orden de rutas y presentacion CLI | todos los servicios | `AtlasOrchestrator` | estable | `test_structured_execution_orchestrator.py`, E2E |
| `core/operational_route_executor.py` | ejecutar RouteDecision tipada | handlers, memoria, tools | `OperationalRouteExecutor`, handlers | estable | `test_operational_route_executor.py` |
| `core/atlas_request_normalizer.py` | normalizar input estructurado | contratos Atlas | normalizer/resultados | experimental | tests Atlas request |
| `core/atlas_request_classifier.py` | clasificar input estructurado | contratos Atlas | classifier/resultados | experimental | tests Atlas request |
| `core/atlas_request_adapter.py` | adaptar a AtlasRouter | contratos Atlas | adapter/resultados | experimental | tests Atlas request |
| `core/atlas_router.py` | rutas CAPABILITY y AGENT | capability service, AgentSystem | `AtlasRouter` | experimental | `test_atlas_router.py` |

## Planning y contratos de plan

| Archivo | Responsabilidad | Dependencias principales | Contratos publicos | Estado | Pruebas |
|---|---|---|---|---|---|
| `core/planner.py` | modelos y generacion de planes | schemas, planners | `ExecutionPlan`, `ExecutionStep`, `Planner` | estable | `test_execution_plan.py`, planner tests |
| `core/deterministic_multi_tool_planner.py` | patrones multi-tool locales | criteria, validator | `DeterministicMultiToolPlanner` | estable | `test_deterministic_multi_tool_planner.py` |
| `core/hybrid_execution_planner.py` | combinar plan local/proveedor | PromptClient, parser | `HybridExecutionPlanner` | estable con proveedor opcional | `test_hybrid_execution_planner.py` |
| `core/execution_plan_validator.py` | validar y firmar | plan, schemas, retry | `ExecutionPlanValidator`, `plan_signature` | estable | `test_execution_plan_validator.py` |
| `core/execution_plan_topology.py` | orden topologico | plan | contratos de topologia | estable | `test_execution_plan_topology.py` |
| `core/execution_plan_registry.py` | planes registrados | plan/signature | `ExecutionPlanRegistry` | estable | `test_execution_plan_registry.py` |
| `core/execution_plan_library.py` | librerias declarativas | registry | contracts de library | experimental | `test_execution_plan_library.py` |
| `core/acceptance_criteria.py` | criterios tipados | stdlib | `AcceptanceCriterion` | estable | verification tests |
| `core/execution_replanner.py` | replan limitado | Planner, Validator | `ExecutionReplanner`, `ReplanPolicy` | estable | `test_execution_replanner.py` |
| `core/historical_plan_adjustment.py` | ajustes conservadores | history, validator | `HistoricalPlanAdjuster` | estable | historical adjustment tests |

## Argumentos, contexto y referencias

| Archivo | Responsabilidad | Dependencias principales | Contratos publicos | Estado | Pruebas |
|---|---|---|---|---|---|
| `core/execution_arguments.py` | mapping inmutable y seguro | references | `ExecutionArguments` | estable | `test_execution_arguments.py` |
| `core/execution_context.py` | resultados, variables y estados | variables, UUID | `ExecutionContext`, snapshot | estable | `test_execution_context.py` |
| `core/parameter_resolver.py` | resolver referencias tipadas | arguments, references | `ParameterResolver` | estable | parameter resolver tests |
| `core/step_output_reference.py` | referencia a output previo | stdlib | `StepOutputReference` | estable | reference tests |
| `core/execution_variable_reference.py` | referencia a variable | stdlib | `ExecutionVariableReference` | estable | variable reference tests |
| `core/execution_variable_binding.py` | binding de outputs | context | binding contracts | estable | binding tests |
| `core/structured_reference_path.py` | paths estructurados acotados | stdlib | parser/path contracts | estable | reference tests |
| `tools/argument_schema.py` | schemas conversacionales | ToolRegistry | schemas y validator | estable | `test_argument_schema.py` |
| `tools/tool_schema.py` | schema de ejecucion por tool | types | `ToolSchema` | estable | `test_tool_schema.py` |

## Gobierno, ejecucion y supervision

| Archivo | Responsabilidad | Dependencias principales | Contratos publicos | Estado | Pruebas |
|---|---|---|---|---|---|
| `core/execution_strategy.py` | seleccionar/validar estrategia | history, report | Strategy contracts | estable | `test_execution_strategy.py` |
| `core/execution_authorization.py` | gate, confirmaciones, despacho | plan/signature | Authorization y Dispatcher | estable | `test_execution_authorization.py` |
| `core/structured_execution.py` | coordinar ciclo estructurado | todo el nucleo ejecutivo | `StructuredExecutionCoordinator` | estable | structured orchestrator tests |
| `core/execution_plan_executor.py` | ejecutar plan y resolver pasos | context, tools, retry | `ExecutionPlanExecutor`, resultados | estable | executor tests |
| `core/execution_supervisor.py` | sesiones y transiciones | persistence contracts | `ExecutionSupervisor`, `ExecutionSession` | estable | supervisor tests |
| `core/execution_retry.py` | politicas de reintento | stdlib | `RetryPolicy`, `RetryEngine` | estable | `test_execution_retry.py` |
| `core/execution_dependency_resolver.py` | dependencias listas | plan/context | resolver contracts | estable | dependency tests |
| `core/execution_condition.py` | condiciones declarativas | references | condition contracts | estable | condition tests |
| `core/execution_resources.py` | seleccion/reserva local | plan | resource contracts | experimental | resource tests |
| `core/execution_priority.py` | prioridad de pasos | plan | priority contracts | experimental | priority tests |
| `core/concurrent_step_executor.py` | concurrencia inyectable | executor/resources | concurrent contracts | experimental | concurrent tests |
| `core/autonomous_execution.py` | fachada autonoma acotada | nucleo compartido | `AutonomousExecutionOrchestrator` | experimental | autonomous tests |
| `core/subplan_executor.py` | subplanes declarativos | executor | `SubplanExecutor` | experimental | subplan tests |

## Verificacion, correccion, informes y persistencia

| Archivo | Responsabilidad | Dependencias principales | Contratos publicos | Estado | Pruebas |
|---|---|---|---|---|---|
| `core/goal_verifier.py` | evaluar evidencia | criteria, plan result | `GoalVerifier`, statuses | estable | `test_objective_outcome_verification.py` |
| `core/objective_correction.py` | correccion determinista limitada | verifier, plan | correction contracts | estable | `test_objective_correction.py` |
| `core/execution_report.py` | informe seguro | supervisor, verifier | `OperationalExecutionReport` | estable | `test_execution_report.py` |
| `core/execution_session_persistence.py` | snapshots JSON atomicos | supervisor, serializers | repository/recovery contracts | estable | `test_execution_session_persistence.py` |
| `core/resumable_execution_store.py` | checkpoint reanudable | executor state | `JsonResumableExecutionStore` | estable | `test_resumable_execution_store.py` |
| `core/execution_history.py` | historial consultable | supervisor, metrics | history contracts | estable | history tests |
| `core/execution_history_advisor.py` | recomendaciones conservadoras | history | advisor contracts | estable | advisor tests |
| `core/execution_observability_serializer.py` | serializacion segura | result contracts | serializers | estable | observability tests |
| `core/execution_observability_deserializer.py` | lectura compatible | serializers | deserializers | estable | observability tests |
| `core/execution_trace.py` | eventos acotados | stdlib | trace contracts | estable | trace tests |

## Herramientas, agentes, memoria y modelos

| Archivo | Responsabilidad | Dependencias principales | Contratos publicos | Estado | Pruebas |
|---|---|---|---|---|---|
| `tools/registry.py` | registro de 40 tools | BaseTool, schema | `ToolRegistry` | estable | registry/catalog tests |
| `tools/executor.py` | invocar tool registrada | registry, context | `ToolExecutor` | estable | executor/tool tests |
| `tools/single_tool_runner.py` | ejecucion simple confirmable | selector, validator | `SingleToolRunner` | legado/estable | runner tests |
| `tools/tool_chain_runner.py` | cadenas lineales | tool executor | `ToolChainRunner` | legado/estable | chain tests |
| `tools/desktop/*` | control Windows | Win32/controller | desktop tools | parcial | desktop tests |
| `tools/filesystem/*` | read/write/list | FileService | filesystem tools | estable | filesystem/E2E |
| `agents/registry.py` | agentes clasicos | BaseAgent | `AgentRegistry` | legado/estable | agent tests |
| `agents/chat_agent.py` | conversacion | PromptClient | `ChatAgent` | estable | conversation tests |
| `agents/coding_agent.py` | codigo y escritura propuesta | file use cases | `CodingAgent` | legado | coding tests |
| `agents/project_agent.py` | analisis determinista de proyecto | AST use cases | `ProjectAgent` | estable | project agent tests |
| `core/agent_system.py` | composicion declarativa | agent/skill services | `AgentSystem` | experimental | `test_agent_system.py` |
| `core/agent_executor.py` | handlers declarativos | registry/context | `AgentExecutor` | experimental | agent executor tests |
| `core/multi_agent.py` | equipo secuencial | AgentExecutor | coordinator/resolver | experimental | multi-agent tests |
| `core/skill_system.py` | skills declarativas | skill services | `SkillSystem` | experimental | `test_skill_system.py` |
| `core/capability_orchestrator.py` | capacidades estructuradas | planner/executor | `CapabilityOrchestrator` | experimental | capability tests |
| `memory/conversation.py` | memoria RAM acotada | operational policy | `ConversationMemory` | estable | conversation/memory tests |
| `memory/operational.py` | politicas y sanitizacion | stdlib | `MemoryPolicy`, entries | estable | operational memory tests |
| `models/prompt_client.py` | cliente Ollama | `ollama` | `PromptClient` | estable | prompt/model tests |
| `models/ollama_client.py` | operaciones Ollama | provider callable | `OllamaClient` | estable | model tests |

## Familias auxiliares

| Patron | Responsabilidad | Estado | Pruebas |
|---|---|---|---|
| `core/agent_*.py` | manifests, discovery, registro, delegacion, cooperacion y supervision de agentes | experimental | tests homonimos |
| `bootstrap/agent_*.py` | factories de agentes declarativos | experimental | tests de bootstrap/sistema |
| `core/skill_*.py` | manifest, discovery, registro, resolver y executor de skills | experimental | skill tests |
| `core/capability_*.py` | resolver, planner, servicio y orquestador de capacidades | experimental | capability tests |
| `use_cases/*.py` | interacciones heredadas, voz, proyecto y escritorio | mixto | tests homonimos |
| `voice/` y use cases de voz | captura, STT, TTS y wake word | parcial | voice/wake-word tests |
| `services/*.py` | acceso local a archivos/proyecto | estable | service/use-case tests |
| `domain/refactoring/*.py` | contratos de refactoring | legado | refactoring tests |
