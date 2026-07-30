# Archivos para compartir

## Regla principal

No compartir el repositorio completo. `.env` esta versionado, por lo que un
clon, bundle o `git archive atlas-base-v1.0` sin exclusiones no es seguro.

## Obligatorios

- `docs/architecture_review/*.md`;
- `VERSION`, `RELEASE.md`, `CHECKLIST_FINAL.md`;
- `requirements.txt`;
- `main.py`, `core/atlas.py`, `core/startup.py`;
- `bootstrap/bootstrap.py`;
- `core/orchestrator.py`;
- `core/request_gateway.py`, `core/operational_request_router.py`,
  `core/router.py`, `core/operational_route_executor.py`;
- planning: planner, planners hibrido/determinista y validator;
- ejecucion: arguments, context, resolver, strategy, authorization,
  structured execution, executor y supervisor;
- verificacion/persistencia: verifier, correction, report, session persistence,
  resumable store e history;
- `tools/registry.py`, `tools/executor.py`, tools filesystem;
- agentes clasicos y contratos del AgentSystem declarativo;
- memoria conversacional/operacional y clientes de modelo;
- pruebas focalizadas listadas en `07_TESTING_EVIDENCE.md`.

## Opcionales

- resto de `core/`, `bootstrap/`, `tools/`, `agents/`, `memory/`, `models/`,
  `services/`, `domain/`, `use_cases/` y `voice/`;
- suite completa `tests/`;
- manual interno `docs/manual/`;
- `README.md` y `docs/ARCHITECTURE.md`, marcados como contexto historico;
- ROADMAP/TASKS solo si se advierte que no prueban implementacion.

## Nunca compartir

- `.env`;
- `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/`;
- `.atlas/` y `logs/`;
- `artifacts/` y screenshots;
- `audio_test.wav`;
- modelos `.onnx` o audio de wake word personalizado;
- archivos generados por el usuario;
- dumps, traces o sesiones reales.

## Sanitizar antes de compartir

- notebooks: outputs, metadata, rutas locales y datos de entrenamiento;
- configuraciones exportadas y variables de entorno;
- logs y reportes, incluso si parecen sanitizados;
- screenshots y audio;
- fixtures nuevos que contengan valores parecidos a credenciales;
- paths absolutos de usuario.

## Tamano aproximado

Medicion del checkout congelado excluyendo `.env`, runtime, notebooks,
artifacts y binarios:

- 374 archivos;
- 5.74 MiB sin comprimir;
- aproximadamente 1.13 MiB en ZIP con deflate.

Los documentos de esta fase agregan una cantidad pequena a ese total.

## Compresion recomendada

1. Crear fuera del repositorio una carpeta vacia.
2. Copiar solo los paths de la lista blanca.
3. Ejecutar un escaneo de secretos sobre esa carpeta.
4. Revisar manualmente el listado.
5. Comprimir la carpeta, no el repositorio.

Ejemplo:

```powershell
Compress-Archive `
  -Path C:\ruta\atlas-architecture-review\* `
  -DestinationPath C:\ruta\atlas-base-v1.0-architecture-review.zip
```

No incluir `.git` ni usar el ZIP como mecanismo de backup. Compartir tambien el
hash SHA-256 del ZIP por un canal separado.
