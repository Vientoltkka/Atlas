# Seguridad y safety

## Modelo de confianza

Atlas es una aplicacion local de un solo usuario. El proceso, filesystem,
modelo local y escritorio se consideran recursos del usuario. No existe
aislamiento multiusuario ni frontera de red distribuida.

## Controles existentes

- RequestGateway limita longitud, metadata, adjuntos y contexto.
- Plan Validator rechaza herramientas, argumentos y referencias invalidas.
- `ExecutionArguments` y manifests rechazan valores no finitos y objetos
  arbitrarios.
- No se usa `eval` ni `exec` en los contratos declarativos.
- Las confirmaciones se vinculan a firma, paso, source y expiracion.
- Dispatcher consume una autorizacion una vez dentro del proceso.
- Escritura, tipeo y hotkeys expuestas requieren confirmacion.
- Recovery revalida plan, schema y estado antes de continuar.
- GoalVerifier y Supervisor no ejecutan herramientas.
- Correccion tiene un ciclo maximo y no es recursiva.
- Informes, memoria y logs aplican sanitizacion de marcadores sensibles.
- Persistencia usa escritura temporal, flush, fsync y `os.replace`.

## Riesgos observados

### Paquete y repositorio

`.env` esta versionado. No se ha inspeccionado su contenido durante esta fase.
Esto impide compartir de forma segura un clon, bundle o `git archive` completo
sin una revision humana y una lista blanca.

Tambien estan versionados:

- `artifacts/screenshots/screenshot_20260714_111825.png`;
- `audio_test.wav`;
- `notebooks/train_atlas_wakeword.ipynb`;
- archivos de prueba manual en la raiz.

Pueden contener datos de entorno o metadata y no forman parte del paquete
obligatorio.

### Ejecucion

- La idempotencia del Dispatcher es memoria local; no ofrece exactly-once
  distribuido ni tras perdida total del proceso.
- Persistencia y logs no estan cifrados.
- Algunas tools de escritorio destructivas no requieren confirmacion en su
  descriptor actual, aunque no tienen schema conversacional directo y aplican
  protecciones internas. Su exposicion futura debe revisarse.
- El proveedor LLM puede producir texto no confiable; el plan local y el
  Validator siguen siendo la frontera antes de herramientas.
- No existe rollback general para efectos externos.

## Secretos y datos privados

Nunca compartir:

- `.env`;
- `.atlas/`;
- `logs/`;
- `.git/`;
- `.venv/`;
- caches;
- screenshots, audio, modelos personalizados;
- snapshots o reportes de ejecuciones reales.

Tests contienen valores ficticios como parte de las pruebas de sanitizacion.
Un revisor debe tratarlos como fixtures, no credenciales.

## Preguntas de seguridad prioritarias

1. Debe persistirse el ledger de autorizaciones y dispatch?
2. Que tools de escritorio deben requerir confirmacion obligatoria?
3. Debe cifrarse la persistencia local o separarse por sensibilidad?
4. Como impedir que prompts, outputs o paths privados lleguen a revisores?
5. Conviene retirar `.env` del historial antes de cualquier publicacion?
