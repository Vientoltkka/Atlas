# Limitaciones conocidas

## Bloqueantes para uso diario

No se identifico un bloqueante reproducible en el flujo de texto congelado.

## Importantes no bloqueantes

1. Rutas superiores diferentes entre CLI estructurada, OperationalRouteExecutor
   y AtlasRouter.
2. Modulos muy grandes: Executor, StructuredExecutionCoordinator,
   OperationalRouteExecutor, Orchestrator y Supervisor.
3. Bootstrap centralizado y costoso de revisar.
4. Idempotencia no persistida del Dispatcher.
5. Persistencia y logs sin cifrado.
6. `.env` versionado impide compartir el repositorio completo.

## Deuda tecnica

- Agentes clasicos y declarativos usan registros distintos.
- El AgentSystem declarativo esta vacio por defecto.
- Infraestructura de skills, multiagente y capabilities excede lo expuesto por
  la CLI diaria.
- Fallback heredado mezcla escritorio, correccion, refactoring y agentes.
- Numerosas dataclasses y serializadores aumentan la superficie de
  compatibilidad.
- No hay una politica unica documentada para declarar una API publica.

## Rendimiento

- Ollama domina la latencia de conversacion.
- Bootstrap construye grafo de arquitectura y componentes de voz en el modo
  texto.
- Los modulos y suites amplios aumentan tiempo de validacion.
- No hay objetivos de rendimiento ni benchmarks estables.

## Seguridad

- `.env`, snapshots, logs y artefactos no son compartibles.
- Persistencia local no cifrada.
- Algunas herramientas de escritorio requieren revisar su politica de
  confirmacion antes de ampliar intents.
- No hay sandbox de proceso para tools.

## Capacidades parciales o no disponibles

- Voz y wake word: parciales y dependientes del entorno.
- GUI, instalador y autoarranque: no disponibles. WhatsApp: canal
  operativo via webhook (texto/audio/voice, imagen con caption,
  documento, ubicacion, contactos y respuestas interactivas).
- Web tool, RAG, embeddings y memoria vectorial: no disponibles.
- Multiusuario, sincronizacion y ejecucion distribuida: no disponibles.
- Rollback general: no disponible.

## Mejora futura

Las mejoras pertenecen a V2 y deben priorizarse despues de la revision:

- definir una unica entrada de alto nivel;
- reducir responsabilidades de modulos grandes sin romper contratos;
- consolidar agentes clasicos/declarativos;
- persistir el ledger de dispatch si se requiere exactly-once;
- aislar secretos y datos operativos;
- lazy composition para modos no usados.
