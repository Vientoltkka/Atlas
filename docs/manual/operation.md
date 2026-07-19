# Operación

manual-id: operation

Propósito: listar comandos reales de arranque, pruebas, validación y uso local.

## Activar entorno en PowerShell

```powershell
.\.venv\Scripts\activate
```

## Arrancar Atlas

```powershell
python -B main.py
```

## Salir

Dentro de Atlas:

```text
salir
```

## Tests

```powershell
python -B -m pytest -q
```

## Estado Git

```powershell
git status
```

Los comandos Git deben ejecutarse fuera de Atlas, en PowerShell. Si Atlas está esperando una confirmación, primero responde o cancela.

## Manual

```powershell
python -B -m tools.atlas_manual list
python -B -m tools.atlas_manual show overview
python -B -m tools.atlas_manual search confirmation
python -B -m tools.validate_atlas_manual
```
