# Flujo de ejecución

manual-id: execution_flow

Propósito: describir el recorrido operativo principal de una petición de texto.

## Recorrido principal

1. `main.py` crea `Atlas`.
2. `Atlas.start()` abre el bucle de texto de `AtlasOrchestrator`.
3. `RequestGateway` crea una `AtlasRequest`.
4. `Router.classify_request()` produce una `RouteDecision`.
5. `StructuredExecutionCoordinator` solicita el `ExecutionPlan` al `Planner`.
6. `ExecutionPlanValidator` valida el plan.
7. `ExecutionHistoryAdvisor` consulta experiencia previa y
   `HistoricalPlanAdjuster` aplica únicamente ajustes permitidos.
8. `ExecutionStrategySelector` selecciona y valida la estrategia.
9. `ExecutionAuthorizationGate` autoriza, bloquea o deja confirmaciones
   pendientes.
10. `ExecutionDispatcher` consume el permiso una sola vez.
11. `ExecutionSupervisor` crea y actualiza la `ExecutionSession`.
12. `ExecutionPlanExecutor` ejecuta los pasos mediante herramientas registradas.
13. La sesión se persiste y `ExecutionReportGenerator` deriva el informe.
14. `AtlasOrchestrator` presenta el mensaje contractual y el informe operativo.

Las peticiones no aplicables al planificador estructurado continúan por los
flujos conversacionales existentes. Las APIs internas de bajo nivel se
conservan para composición y compatibilidad, pero no son el comando principal.

## Confirmaciones y bloqueos

- Una confirmación pendiente crea sesión e informe, pero no llega al Executor.
- La revisión manual impide el despacho automático.
- Una autorización consumida no puede despacharse otra vez.
- Dos peticiones nuevas equivalentes reciben permisos distintos; la
  idempotencia se limita a cada solicitud de ejecución.

## Errores

Los errores de herramienta se registran en la sesión y aparecen en el informe.
Los reintentos respetan la política del plan y `NO_RETRY` permanece en un solo
intento. Un fallo controlado no cierra el bucle de texto.
