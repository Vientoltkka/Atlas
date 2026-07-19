# Diagnóstico

manual-id: troubleshooting

Propósito: recopilar problemas reales conocidos, comprobaciones y soluciones seguras.

## Atlas captura comandos de PowerShell

- Síntoma: escribes `git status` o `pytest` y Atlas lo interpreta como respuesta.
- Causa probable: Atlas está dentro del bucle conversacional o esperando confirmación.
- Comprobación: mira si la consola muestra una pregunta de confirmación o el prompt de Atlas.
- Solución segura: responde `no`, cancela o escribe `salir`; luego ejecuta el comando en PowerShell.
- Escalar: si no hay forma de salir sin cerrar la terminal.

## Rutas mal extraídas al modificar confirmaciones

- Síntoma: Atlas propone escribir en una ruta distinta a la deseada.
- Causa probable: frase ambigua o extracción determinista insuficiente.
- Comprobación: inspecciona la operación pendiente antes de confirmar.
- Solución segura: modifica con una frase explícita o cancela y repite.
- Escalar: si una ruta clara y simple se extrae mal de forma reproducible.

## Archivos temporales durante pruebas

- Síntoma: aparecen archivos de prueba o cachés.
- Causa probable: tests que escriben fixtures, capturas o cachés.
- Comprobación: revisar `git status`.
- Solución segura: no borrar cambios ajenos; identificar si son generados por la prueba.
- Escalar: si un test deja archivos no aislados fuera de `tmp_path`.

## Wake word irregular

- Síntoma: Atlas no despierta siempre por voz.
- Causa probable: modelo wake word ausente, fallback STT, micrófono o ruido.
- Comprobación: ejecutar diagnóstico de audio o assistant con logs.
- Solución segura: probar texto primero y revisar configuración de micrófono.
- Escalar: si el dispositivo recibe señal y aun así falla siempre.

## Reconocimiento de voz intermitente

- Síntoma: Atlas escucha pero transcribe vacío o incorrecto.
- Causa probable: nivel RMS bajo, dispositivo equivocado o timeout corto.
- Comprobación: revisar diagnósticos de captura.
- Solución segura: seleccionar entrada correcta y hablar tras activación.
- Escalar: si la captura WAV contiene voz clara y STT falla.

## TTS robótico o lento

- Síntoma: voz artificial, lenta o bloqueante.
- Causa probable: motor local de TTS y configuración del sistema.
- Comprobación: probar respuesta de texto equivalente.
- Solución segura: usar modo texto para operaciones críticas.
- Escalar: si TTS bloquea el bucle.

## Fecha u hora incorrectas

- Síntoma: una respuesta da hora o fecha desactualizada.
- Causa probable: respuesta de modelo sin herramienta temporal.
- Comprobación: comparar con el sistema operativo.
- Solución segura: usar una herramienta adecuada cuando exista; en 5.5A no hay herramienta de hora registrada.
- Escalar: si se documenta como capacidad implementada sin herramienta real.

## Diferencias entre consola y voz

- Síntoma: texto funciona pero voz no.
- Causa probable: la ruta de voz añade STT, wake word y TTS.
- Comprobación: reproducir primero en `python -B main.py`.
- Solución segura: validar por consola antes de depurar voz.
- Escalar: si ambos fallan con la misma petición.

## Unicode en Windows

- Síntoma: caracteres acentuados o árbol visual se ven como escapes.
- Causa probable: página de códigos de consola.
- Comprobación: comparar archivo real con salida de consola.
- Solución segura: conservar UTF-8 en archivos y usar salidas seguras.
- Escalar: si se corrompe el archivo interno, no solo la visualización.
