# Archivos para compartir

## Regla principal

No compartir el repositorio completo ni construir el paquete con `git archive`,
un clon o un bundle. El unico artefacto autorizado es el ZIP generado desde la
lista blanca cerrada de `scripts/build_architecture_review_package.py`.

## Comando oficial

Desde la raiz del repositorio:

```powershell
python -B scripts/build_architecture_review_package.py
```

El comando genera:

- `dist/review/atlas-architecture-review-v1.0.zip`;
- `dist/review/atlas-architecture-review-v1.0.manifest.json`.

Ambos archivos son locales, estan ignorados por Git y deben eliminarse despues
de completar la revision externa.

## Lista blanca incluida

- los doce documentos de `docs/architecture_review/`;
- `.env.example`, `VERSION`, `RELEASE.md`, `CHECKLIST_FINAL.md`,
  `requirements.txt`, `README.md` y `main.py`;
- `docs/ARCHITECTURE.md`, `docs/execution_decision.md` y `docs/manual/*.md`;
- codigo Python de `agents/`, `api/`, `bootstrap/`, `core/`, `domain/`,
  `memory/`, `models/`, `scripts/`, `services/`, `tools/`, `use_cases/` y
  `voice/`;
- las pruebas focalizadas declaradas en `AUTHORIZED_TEST_FILES`.

`MANIFEST.json`, incluido dentro del ZIP, enumera de forma exacta cada path,
tamano y hash SHA-256. El manifest adyacente debe coincidir byte a byte con el
interno.

## Exclusiones obligatorias

- `.env`, `.git/`, `.venv/`, `venv/` y caches;
- `.atlas/`, `logs/`, sesiones, trazas y estado de ejecucion;
- `artifacts/`, screenshots, notebooks y archivos generados por el usuario;
- audio, imagenes, modelos `.onnx`, ZIP previos y temporales;
- cualquier path absoluto, traversal, symlink, duplicado o archivo fuera de la
  lista blanca;
- cualquier archivo que active el escaneo de secretos de alta confianza.

## Verificacion antes de entregar

1. Abrir el ZIP y comprobar que no esta corrupto.
2. Comparar `MANIFEST.json` interno con el manifest adyacente.
3. Recalcular SHA-256 y tamano de cada entrada contra el manifest.
4. Revisar el listado completo y confirmar que no contiene categorias
   excluidas.
5. Ejecutar de nuevo el generador y confirmar el mismo orden y contenido logico.
6. Compartir el hash SHA-256 del ZIP por un canal separado.

## Historial y respuesta ante incidentes

La retirada de `.env` del indice actual no elimina versiones historicas. Antes
de publicar el repositorio o su historial, auditar todos los commits. Si se
detecta una credencial real, revocarla o rotarla primero y sanear el historial
solo mediante un procedimiento separado y revisado. No reescribir historial
durante la construccion de este paquete.