# Persistencia y recuperacion

## Almacenes

| Almacen | Ruta por defecto | Contenido |
|---|---|---|
| Sesiones supervisadas | `.atlas/execution_sessions/*.json` | plan, pasos, outputs seguros, eventos, estrategia y autorizacion |
| Estado reanudable | `.atlas/execution_state.json` | checkpoint del executor/coordinador |
| Log operativo | `logs/atlas.log` | inicio, cierre y errores sanitizados |
| Memoria conversacional | RAM | mensajes y contexto reciente |

## Escritura

`FileExecutionSessionRepository` serializa JSON determinista, limita el tamano,
escribe un archivo temporal, hace flush/fsync y reemplaza atomicamente el
destino. El lock es por `session_id` dentro del proceso.

`ExecutionSupervisor` continua el contador a partir de sesiones persistidas,
evitando sobrescribir IDs tras reinicio. La suite prueba IDs unicos, escritura
concurrente local y preservacion del archivo previo ante fallo.

## Lectura y compatibilidad

Los snapshots tienen `schema_version`. Payloads corruptos o versiones no
soportadas producen errores tipados. Campos introducidos en fases posteriores
tienen defaults legacy donde esta documentado y probado.

## Recuperacion

`ExecutionRecoveryService`:

1. lista snapshots;
2. valida schema, plan y dependencias;
3. restaura la sesion en Supervisor;
4. clasifica con `ExecutionRecoveryPolicy`;
5. no ejecuta durante discovery/recovery.

Decisiones posibles:

- terminal: no reanudar;
- waiting confirmation: preservar confirmacion;
- running/interrupted/active batch ambiguo: revision manual;
- pending recovery-safe: reanudacion permitida.

## Reanudacion

El Executor restaura `ExecutionContext`, resultados y variables. Revalida firma
del plan, schema actual y referencias. Los pasos completados no vuelven a
ejecutarse. Los pasos pendientes continúan solo si el estado es coherente.

## Limitaciones

- JSON local sin cifrado.
- Locks e idempotencia limitados al proceso/equipo.
- No hay transaccion conjunta entre log, snapshot y efectos externos.
- Un crash despues de un efecto externo pero antes del snapshot puede requerir
  revision manual.
- El cleanup y retencion de sesiones no tienen politica operativa avanzada.
- Los outputs persistidos pueden contener datos del usuario pese a la
  sanitizacion; no deben compartirse.

## Evidencia

Pruebas principales:

- `tests/test_execution_session_persistence.py`;
- `tests/test_resumable_execution_store.py`;
- `tests/test_structured_execution_orchestrator.py`;
- `tests/test_objective_outcome_verification.py`;
- `tests/test_objective_correction.py`;
- `tests/test_operational_end_to_end.py`.
