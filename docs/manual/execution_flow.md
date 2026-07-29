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
13. `GoalVerifier` separa el estado técnico del cumplimiento del objetivo y
    evalúa únicamente criterios declarados con resultados y contexto reales.
14. La sesión persiste el resultado sanitizado y `ExecutionReportGenerator`
    deriva el informe.
15. `AtlasOrchestrator` presenta el mensaje contractual y el informe operativo.

Las peticiones no aplicables al planificador estructurado continúan por los
flujos conversacionales existentes. Las APIs internas de bajo nivel se
conservan para composición y compatibilidad, pero no son el comando principal.

## Datos entre pasos

Cada `ExecutionStep` conserva sus argumentos en `ExecutionArguments`. Los
valores literales se mantienen sin cambios y una `StepOutputReference` permite
que un paso posterior consuma la salida de otro paso ya completado. Antes de
invocar la herramienta, `ParameterResolver` resuelve esas referencias contra
el `ExecutionContext`; una referencia inexistente, futura o circular se rechaza
sin ejecutar código dinámico.

El resultado resuelto se conserva solo durante la ejecución. La sesión y el
informe registran las claves de argumentos y los identificadores de referencia,
pero no duplican sus valores, para evitar exponer contenido o secretos.

## Verificación del objetivo

Un estado técnico `completed` no implica por sí solo que el objetivo esté
cumplido. Los planes pueden declarar criterios deterministas sobre pasos,
salidas, herramientas, confirmaciones y recursos producidos. El resultado usa
los estados `VERIFIED`, `PARTIALLY_VERIFIED`, `NOT_VERIFIED`, `INCONCLUSIVE`,
`USER_ACTION_REQUIRED` y `NOT_APPLICABLE`.

Los planes antiguos sin criterios continúan ejecutándose, pero su objetivo se
clasifica como `INCONCLUSIVE`; Atlas no inventa evidencia ni los presenta como
verificados.

## Corrección controlada del objetivo

Una verificación `NOT_VERIFIED` o `PARTIALLY_VERIFIED` se clasifica antes de
cualquier acción. Atlas solo prepara una corrección cuando existe un criterio
fallido, evidencia concreta, un valor esperado ya demostrado y un único
recurso declarado. `INCONCLUSIVE` no inicia reparaciones especulativas y
`USER_ACTION_REQUIRED` se detiene.

La primera capacidad correctiva es deliberadamente estrecha:
`RESOURCE_CONTENT_EQUALS` puede producir un fragmento de dos pasos
`write_file` → `read_file`. El valor preservado se suministra mediante una
referencia tipada del `ExecutionContext`; no se inventa ni se solicita a un
modelo. El fragmento conserva el objetivo, pasa por Validator, estrategia,
AuthorizationGate y Dispatcher, y requiere una confirmación nueva ligada a su
propia firma.

Después del único ciclo permitido, `GoalVerifier` vuelve a evaluar el plan
original combinando evidencia previa válida y la nueva lectura. El informe
mantiene visible el `NOT_VERIFIED` inicial y distingue
`VERIFIED_AFTER_CORRECTION`, fallo correctivo y `LIMIT_REACHED`.

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
