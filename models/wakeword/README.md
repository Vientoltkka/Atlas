# Modelo wake word Atlas

Este directorio esta reservado para el modelo personalizado `Atlas.onnx`.

No crees ni renombres archivos simulados como `Atlas.onnx`. El archivo solo es valido si fue generado o descargado mediante un procedimiento compatible con openWakeWord y puede cargarse con la version instalada en el entorno de Atlas.

## Ubicacion

Coloca el modelo real aqui:

```text
models/wakeword/Atlas.onnx
```

Configura la ruta en `.env`:

```text
ATLAS_WAKE_WORD_MODEL_PATH=models/wakeword/Atlas.onnx
ATLAS_WAKE_WORD_SENSITIVITY=0.55
```

## Requisitos de audio

openWakeWord espera audio PCM mono de 16 bits a 16 kHz. Atlas procesa frames de 1280 muestras, equivalentes a 80 ms.

## Validacion

Antes de usar el modo de voz completo, valida solo la wake word:

```powershell
python scripts/test_wake_word.py --list-microphones
python scripts/test_wake_word.py --microphone 0
```

El script debe cargar `Atlas.onnx`, abrir el microfono seleccionado, mostrar puntuaciones y marcar la deteccion cuando digas "Atlas".

## Git

Los modelos `.onnx` deben permanecer fuera de Git salvo decision expresa. La regla `*.onnx` en `.gitignore` evita versionarlos por accidente.
