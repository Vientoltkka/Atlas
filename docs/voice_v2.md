# Voz V2 — operación y diagnóstico (FASE 17.1)

## Comando oficial

Desde la raíz del repositorio:

```powershell
python -B main.py --voice
```

`--voice` inicia conversación continua sin wake word. `--assistant` es una ruta heredada independiente con activación; no es el comando oficial de Voz V2 en esta fase. El arranque sin argumentos conserva el modo texto.

## Flujo real reutilizado

```text
Micrófono (sounddevice)
→ SoundDeviceAudioCapture: selección, calibración, VAD por RMS y fin por silencio
→ SpeechEngineUseCase
→ FasterWhisperSpeechToTextProvider (faster-whisper local)
→ normalización y filtro de transcripción
→ AtlasOrchestrator.process_voice_prompt()
→ router/herramientas/modelo normales de Atlas
→ respuesta escrita en consola
→ Pyttsx3SpeechOutputEngine (SAPI5 local)
→ reproducción bloqueante de la respuesta
→ liberación de la captura del turno
→ nueva escucha
```

Punto de entrada: `main.py -> Atlas.start_voice() -> AtlasOrchestrator.start_voice() -> VoiceConversationUseCase.execute_manual()`.

No se ha creado un segundo orquestador, memoria, Planner ni stack de voz. El contexto legítimo se conserva en la memoria normal del mismo `AtlasOrchestrator`; audio y transcripciones temporales pertenecen a cada turno/sesión y no se persisten por Voz V2.

## Componentes y dependencias

| Etapa | Implementación | Disponibilidad |
|---|---|---|
| Captura/VAD | `numpy` + `sounddevice` | Obligatoria para `--voice`; no bloquea el modo texto |
| STT | `faster-whisper`, modelo `small`, CPU/int8, español | Obligatoria para transcribir; carga diferida |
| Procesamiento | `AtlasOrchestrator.process_voice_prompt()` | Misma ruta funcional que Atlas texto |
| Modelo | Ollama/modelo seleccionado por el runtime normal | Necesario para consultas que lleguen al modelo |
| TTS/reproducción | `pyttsx3` + SAPI5 | Opcional: si falla, la sesión continúa por texto en `DEGRADED` |
| Wake word | `openwakeword` o fallback STT | No participa en `--voice`; solo en `--assistant` |

La ausencia de una dependencia de voz se comunica sin traceback visible. Para instalar el conjunto declarado:

```powershell
python -m pip install -r requirements.txt
```

## Micrófonos

Listar dispositivos, host API y capacidad de apertura:

```powershell
python -B main.py --list-microphones
```

Probar tres segundos un índice sin cargar Atlas, Whisper ni TTS:

```powershell
python -B main.py --test-microphone 1
```

Seleccionar el dispositivo con `ATLAS_MICROPHONE_INDEX`. Si no se define, se usa el predeterminado seguro. La captura evita WDM-KS incompatible con el flujo bloqueante y puede pasar a un dispositivo MME, DirectSound o WASAPI con señal real.

## Configuración

Todas son opcionales y sus ejemplos permanecen vacíos en `.env.example`.

| Variable | Default | Función |
|---|---:|---|
| `ATLAS_MICROPHONE_INDEX` | dispositivo del sistema | Entrada de audio |
| `ATLAS_VOICE_SAMPLE_RATE` | `16000` | Frecuencia de captura (8000–48000 Hz) |
| `ATLAS_VOICE_MAX_DURATION` | `8.0` | Duración máxima de frase |
| `ATLAS_VOICE_INITIAL_SILENCE_TIMEOUT` | `3.0` | Espera máxima de voz inicial |
| `ATLAS_VOICE_TRAILING_SILENCE` | `0.75` | Silencio que cierra la frase |
| `ATLAS_VOICE_RMS_THRESHOLD` | `0.004` | Umbral base de voz; se calibra por sesión |
| `ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES` | `3` | Pruebas consecutivas sin señal antes de buscar otro dispositivo |
| `ATLAS_STT_MODEL` | `small` | Modelo faster-whisper |
| `ATLAS_STT_LANGUAGE` | `es` | Idioma STT |
| `ATLAS_STT_MIN_CONFIDENCE` | `0.35` | Filtro de baja confianza |
| `ATLAS_STT_MAX_NO_SPEECH_PROBABILITY` | `0.65` | Filtro de silencio |
| `ATLAS_OLLAMA_TIMEOUT` | `120` | Timeout nativo de la petición HTTP al servidor Ollama |
| ATLAS_OLLAMA_KEEP_ALIVE | 10m | Mantiene el modelo cargado entre turnos; Ollama informa carga y generación reales |
| `ATLAS_VOICE_MODEL_TIMEOUT` | `135` | Supervisor externo de respuesta del modelo |
| `ATLAS_VOICE_MAX_CONSECUTIVE_TIMEOUTS` | `2` | Límite que cierra una sesión con workers reiteradamente agotados |
| `ATLAS_TTS_VOICE` | voz española disponible | ID o nombre parcial de voz SAPI5 |
| `ATLAS_TTS_RATE` | `175` | Velocidad TTS |
| `ATLAS_TTS_VOLUME` | `1.0` | Volumen TTS (0–1) |
| `ATLAS_VOICE_METRICS` | desactivado | Resumen seguro por turno |
| `ATLAS_VOICE_DIAGNOSTICS` | desactivado | Diagnóstico operativo ampliado |
| `ATLAS_VOICE_DEBUG` | desactivado | Trazas técnicas de desarrollo |

## Estados

Cada sesión conserva internamente el historial de estados:

`STARTING -> READY -> LISTENING -> TRANSCRIBING -> PROCESSING -> SPEAKING -> READY`

Los errores recuperables pasan por `RECOVERING`; una capacidad opcional ausente o fallida usa `DEGRADED`. Todo cierre termina en `STOPPING -> STOPPED`. Sin barge-in no se abre una nueva captura mientras TTS está reproduciendo.

Órdenes habladas o escritas mínimas de cierre: `salir`, `exit`, `quit`. También se conserva la política previa (`terminar`, `cancelar`, `adiós`, `stop`). Ctrl+C produce cierre voluntario y liberación de recursos.

## Métricas

Con `ATLAS_VOICE_METRICS=1`, cada turno muestra valores procedentes del reloj monotónico real:

```text
Inicio de voz: X ms
Captura: X ms
STT: X ms
Atlas: X ms
Modelo: X ms
Síntesis TTS: X ms
Reproducción: X ms
Total: X ms
```

`Síntesis TTS` mide preparación del motor y encolado de la frase; `Reproducción` mide `runAndWait()`. `Atlas` mide el procesamiento completo posterior al STT; `Modelo` es el subconjunto de ese tiempo cuando la ruta llama al modelo. `Total` es tiempo de pared del turno y no se obtiene sumando campos solapados. Las métricas no contienen audio, texto transcrito, respuestas, tokens, claves ni secretos.
Con métricas activadas, cada respuesta no streaming de Ollama añade una línea `[ollama-metrics]` con modelo real, carga, generación, total, reutilización observada y `keep_alive`. `modelo_reutilizado=si` exige que el mismo cliente ya haya usado ese modelo y que `load_duration` sea inferior a un segundo; se conserva también el valor bruto de carga para no ocultar recargas.

## Recuperación y errores frecuentes

- Silencio o transcripción vacía: informa del fallo, pasa por `RECOVERING` y vuelve a `LISTENING`.
- Una captura completa silenciosa no cambia de micrófono. Tras una transcripción válida, Voz V2 abre directamente un stream fresco sobre el mismo dispositivo, sin probe previo ni descarte del primer bloque. Solo busca fallback tras `ATLAS_VOICE_MICROPHONE_FALLBACK_FAILURES` capturas completas sin voz o ante un fallo real de apertura; STT válido reinicia todas las penalizaciones temporales.
- Micrófono o índice inválido: usar `--list-microphones` y `--test-microphone`; un error de dispositivo es crítico para la sesión de voz, no para texto.
- `faster-whisper` ausente/modelo no disponible: instalar requisitos o corregir `ATLAS_STT_MODEL`; no se imprime traceback.
- Ollama detenido o modelo ausente produce `model_failure`; timeout produce `model_timeout`; respuesta vacía produce `empty_response`. Un timeout aislado es recuperable, pero el límite consecutivo evita acumular workers bloqueados. El timeout nativo de Ollama debe ser menor que el supervisor de voz.
- `pyttsx3`/SAPI5 ausente o reproducción fallida: estado `DEGRADED`; la respuesta escrita y los siguientes turnos continúan.
- EOF/Ctrl+C/cierre hablado: la salida atraviesa `STOPPING` y `STOPPED` y cierra captura/TTS.

## Protocolo físico de cierre

1. Ejecutar `python -B main.py --list-microphones`.
2. Probar el índice elegido con `python -B main.py --test-microphone <índice>`.
3. Activar `ATLAS_VOICE_METRICS=1` y arrancar `python -B main.py --voice`.
4. Hacer dos preguntas relacionadas, una lectura segura, guardar silencio, hacer otra petición válida y completar al menos cinco turnos.
5. Decir `salir`.
6. Confirmar voz audible, retorno automático a escucha, ausencia de traceback, salida 0 y disponibilidad inmediata del micrófono para otra aplicación.

No debe registrarse una simulación como prueba física.

## Límites para FASE 17.2

Pendiente expresamente: barge-in durante TTS, cancelación de TTS por voz, escucha simultánea durante reproducción, reducción avanzada de latencia, streaming parcial STT/LLM/TTS y wake word robusta. Voz V2 actual es half-duplex: primero escucha y después habla.