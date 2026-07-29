# Limitaciones

manual-id: limitations

Propósito: registrar límites actuales para evitar sobreprometer capacidades.

## Límites operativos actuales

- La planificación estructurada local es determinista; el proveedor de planes
  basado en modelo continúa siendo opcional.
- Atlas no infiere criterios de aceptación semánticos. Cuando el plan no
  declara evidencia determinista suficiente, el objetivo queda `INCONCLUSIVE`
  aunque la ejecución técnica haya terminado.
- Los ajustes históricos están limitados por políticas explícitas y no
  constituyen aprendizaje automático.
- La idempotencia del dispatcher es local al proceso y al permiso de ejecución;
  no existe coordinación distribuida.
- Una confirmación pendiente debe confirmarse o cancelarse antes de iniciar
  otro plan estructurado.
- No hay rollback general automático para efectos externos.
- La concurrencia solo se activa cuando una política y un ejecutor concurrente
  compatibles se inyectan explícitamente.
- Los fallos de planificación que no producen ningún `ExecutionPlan` no crean
  una `ExecutionSession`; sí devuelven un error estructurado controlado.
- La corrección del objetivo no es semántica ni general. En esta fase solo
  repara de forma determinista un único `RESOURCE_CONTENT_EQUALS` cuando el
  valor correcto ya existe en el contexto de ejecución.
- No hay corrección recursiva: el límite predeterminado es un ciclo, tres
  pasos, un recurso, una confirmación nueva y una reverificación.
- `INCONCLUSIVE`, recursos no declarados, valores ausentes, cambios de objetivo
  y herramientas no autorizadas nunca inician una corrección automática.

## Límites de herramientas

- Muchas herramientas de escritorio están registradas, pero no todas tienen un
  intent conversacional o schema completo.
- No hay herramienta web registrada.
- No hay herramienta de eliminación de archivos registrada.
- `read_file` y `write_file` operan como texto UTF-8. Conservan el contenido
  lógico, pero no garantizan una copia binaria idéntica cuando el origen mezcla
  convenciones de saltos de línea.

## Voz

La voz conserva sus rutas actuales y solo se considera smoke-tested en esta
fase. La optimización profunda de captura, wake word y latencia queda fuera del
alcance de la integración operativa por texto.

## Documentación heredada

`ROADMAP.md`, `TASKS.md` y `docs/ARCHITECTURE.md` contienen objetivos o
aspiraciones que no siempre reflejan el estado real. Este manual actúa como
capa de estado operativo validado.
