# Atlas

Atlas es un sistema operativo de IA personal basado en agentes.

## Objetivos

- Orquestar múltiples agentes especializados.
- Utilizar modelos locales y APIs.
- Memoria a corto y largo plazo.
- Control del ordenador.
- Voz.
- WhatsApp.
- Programación.
- Investigación.
- CrossFit / HYROX.
- Nutrición.
- Finanzas.

## Arquitectura

Atlas Core
│
├── Router
├── Memory
├── Agents
├── Models
├── Tools
└── Voice

Versión inicial:
Atlas v1.0

## Arranque manual en Windows

Desde `C:\AI\Atlas`, usa siempre el Python del entorno virtual del proyecto. Esto evita ejecutar Atlas accidentalmente con otro Python instalado en Windows y perder dependencias del proyecto.

Chat / interfaz:

```powershell
.\.venv\Scripts\python.exe -B main.py --chat
```

Voz:

```powershell
.\.venv\Scripts\python.exe -B main.py --voice
```

No uses `python main.py ...` como comando operativo salvo que el entorno virtual ya esté activado y hayas comprobado que `python` apunta a `C:\AI\Atlas\.venv\Scripts\python.exe`.

## Voz V1

Consulta [operación, configuración, métricas, límites conocidos y objetivo Voz V2](docs/voice_v2.md).
