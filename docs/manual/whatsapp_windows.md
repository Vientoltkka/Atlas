# WhatsApp en Windows: servicio operativo (V4.1-W2)

## Mecanismo elegido

La tarea programada de Windows (Task Scheduler) es el mecanismo nativo
elegido frente a un servicio Windows con pywin32:

- no anade dependencias nuevas;
- no requiere privilegios de administrador para una tarea de inicio de
  sesion del usuario actual;
- ofrece reinicio automatico ante fallo y politica de instancia unica.

## Comando oficial ejecutado

La tarea ejecuta exactamente:

```
python -B main.py --whatsapp-webhook
```

con directorio de trabajo igual a la raiz del proyecto. El interprete
python registrado es el mismo usado durante la instalacion
(`sys.executable`).

## Instalacion

1. Crea o revisa el archivo `.env` con las variables obligatorias:
   `ATLAS_WHATSAPP_VERIFY_TOKEN`, `ATLAS_WHATSAPP_ACCESS_TOKEN`,
   `ATLAS_WHATSAPP_PHONE_NUMBER_ID` (ver `.env.example`). Los valores
   nunca se escriben en la tarea ni en logs.
2. Ejecuta desde la raiz del proyecto:

```
python -B core/windows_task.py install
```

La instalacion falla de forma controlada si faltan credenciales,
mostrando solo los nombres de las variables que faltan.

## Arranque automatico

La tarea se dispara al iniciar sesion (LogonTrigger). Para arrancarla
manualmente sin esperar al proximo inicio de sesion:

```
python -B core/windows_task.py start
```

## Recuperacion ante fallo

Si el proceso termina inesperadamente, Task Scheduler lo relanza hasta
3 veces con un intervalo de 1 minuto. El limite de tiempo de ejecucion
esta desactivado (`ExecutionTimeLimit=PT0S`) para procesos de larga
duracion, y `MultipleInstancesPolicy=IgnoreNew` evita instancias
duplicadas.

## Parada y desinstalacion

Parada de la tarea (termina el proceso):

```
python -B core/windows_task.py stop
```

Nota: `schtasks /End` termina el proceso de forma inmediata; no hay
cierre graceful a traves de la tarea. Para un cierre limpio, detén el
proceso interactivo con Ctrl+C.

Eliminacion de la tarea:

```
python -B core/windows_task.py uninstall
```

## Estado y diagnostico

Estado de la tarea:

```
python -B core/windows_task.py status
```

Diagnostico del canal (webhook en marcha):

- `GET http://127.0.0.1:<puerto>/health` -> 200 HEALTHY/DEGRADED,
  503 UNHEALTHY.
- `GET http://127.0.0.1:<puerto>/metrics` con cabecera
  `Authorization: Bearer <ATLAS_WHATSAPP_VERIFY_TOKEN>` -> contadores.

El puerto por defecto es 8000 (`ATLAS_WHATSAPP_WEBHOOK_PORT`).
Los registros operativos se escriben en `logs\atlas.log`.

## Actualizacion

Tras actualizar el codigo, no hace falta reinstalar la tarea salvo que
cambie la ruta del proyecto o el interprete Python:

```
python -B core/windows_task.py stop
git pull
python -B core/windows_task.py start
```
