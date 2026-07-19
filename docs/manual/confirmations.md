# Confirmaciones

manual-id: confirmations

Propósito: documentar acciones peligrosas, respuestas aceptadas y modificación segura.

## Acciones que requieren confirmación

Según el registry actual:

- `write_file`
- `desktop.type_text`
- `desktop.press_hotkey`

## Respuestas aceptadas

Confirmar: `s`, `si`, `sí`, `yes`, `y`.  
Rechazar: `n`, `no`.

Una respuesta ambigua no ejecuta nada y mantiene la confirmación pendiente.

## Cancelación

Si el usuario rechaza, Atlas devuelve estado `CANCELLED` y no ejecuta la acción pendiente. En cadenas, los pasos ya completados se conservan como hechos históricos; el paso pendiente no se ejecuta.

## Modificación segura

Durante una confirmación, el usuario puede inspeccionar o modificar argumentos soportados. Atlas revalida la operación y emite un nuevo `confirmation_id` interno. El id anterior queda invalidado.

## Cadenas pausadas

Si una cadena `read -> write` ya leyó el origen y está esperando confirmar escritura, Atlas no permite cambiar retroactivamente el archivo leído. Para cambiar el origen hay que cancelar y crear una operación nueva.

## Advertencia operativa

Mientras Atlas espera confirmación, la siguiente entrada pertenece a Atlas. No escribas comandos de PowerShell dentro de esa espera; cancela o responde antes.
