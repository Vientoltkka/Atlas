# Checklist final de Atlas Base v1.0

Objetivo: validar la base en menos de cinco minutos desde `C:\AI\Atlas`.

## 1. Comprobacion automatica rapida

```powershell
python -B -m py_compile main.py core\orchestrator.py bootstrap\bootstrap.py
python -B -m pytest tests/test_windows_startup.py tests/test_operational_end_to_end.py tests/test_operational_multi_step_end_to_end.py -q -p no:cacheprovider
git diff --check
```

Resultado esperado: todos los comandos terminan con codigo 0.

## 2. Arranque y conversacion

```powershell
python -B main.py
```

Comprobar:

- [ ] El banner muestra `Atlas - Base operativa v1.0`.
- [ ] El estado es `preparado` y el modo es `texto`.
- [ ] `¿Que herramientas tienes?` devuelve el catalogo real.
- [ ] `Lee README.md` muestra su contenido sin traceback.
- [ ] `¿Cual es la capital de Francia?` devuelve Paris.

## 3. Confirmacion y cancelacion

Ejecutar:

```text
Escribe validacion en .atlas/checklist_v1.txt
confirmo
Escribe cancelar en .atlas/checklist_cancelado_v1.txt
cancela
salir
```

Comprobar:

- [ ] La primera escritura espera confirmacion y se ejecuta una vez.
- [ ] La segunda escritura se cancela y no crea el archivo.
- [ ] `salir` muestra `Hasta pronto.` y cierra sin traceback.

Eliminar `.atlas/checklist_v1.txt` despues de comprobar su contenido. El
archivo cancelado no debe existir.

## 4. Criterio final

- [ ] Arranque correcto.
- [ ] Conversacion basica funcional.
- [ ] Lectura funcional.
- [ ] Confirmacion y cancelacion seguras.
- [ ] Cierre limpio.
- [ ] Pruebas rapidas y `git diff --check` correctos.

Si cualquier casilla falla, Atlas Base v1.0 deja de considerarse validada hasta
resolver la incidencia.
