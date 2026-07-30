# Evidencia de pruebas

## Alcance de la suite

El repositorio contiene 134 archivos `test_*.py` y aproximadamente 2590
funciones de prueba declaradas. Las parametrizaciones elevan el total ejecutado.

## Comandos de congelacion

```powershell
$files = @(rg --files -g '*.py')
python -B -m py_compile $files
python -B -m compileall -q .
python -B -m pytest -q -p no:cacheprovider --basetemp C:\pytest-fase-16-1-full
git diff --check
python -B main.py
```

## Evidencia base FASE 15.8

- Suite completa: 2897 passed, 1 skipped.
- CLI fisica: banner, conversacion, catalogo, lectura, pregunta general,
  confirmacion, cancelacion y cierre.
- Sesiones previas preservadas y sin sobrescritura.

## Evidencia focalizada por riesgo

| Riesgo | Pruebas |
|---|---|
| routing y RouteDecision | request router, route executor, operational E2E |
| argumentos/contexto/referencias | execution arguments/context/parameter resolver |
| autorizacion y doble despacho | `test_execution_authorization.py` |
| `NO_RETRY` | retry, validator, authorization y correction |
| persistencia/reanudacion | session persistence, resumable store, structured orchestrator |
| objetivo verificado/no verificado | objective outcome verification |
| correccion limitada | objective correction |
| sanitizacion | reports, memory, manifests, handlers y startup |
| retrocompatibilidad | serializers, persistence, report y fases anteriores |
| arranque Windows | `test_windows_startup.py` |

## Resultado FASE 16.1

- `py_compile` sobre todos los archivos Python: codigo 0.
- `compileall -q .`: codigo 0; aviso no bloqueante al listar
  `.pytest_cache`.
- Pruebas end-to-end y de seguridad focalizadas: 147 passed en 69.67 s.
- Suite completa: 2897 passed, 1 skipped en 235.12 s.
- Validador del manual: `Manual valido.`
- `git diff --check`: codigo 0.
- Arranque fisico `python -B main.py`: codigo 0 en 10.17 s.
- Banner `Atlas - Base operativa v1.0`, estado preparado, modo texto y cierre
  `Hasta pronto.` verificados.
- No se mostro ningun traceback.

## Limites de la evidencia

- La voz depende de hardware y tiene cobertura principalmente fake/smoke.
- No hay pruebas distribuidas ni multiusuario.
- Los tests no demuestran exactly-once frente a crash entre efecto externo y
  persistencia.
- La pregunta general depende del modelo local y su latencia.
