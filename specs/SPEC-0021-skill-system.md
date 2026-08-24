# SPEC-0021 — Skill System (V3.2.1)

## Estado

IMPLEMENTADO (consolidación mínima; el subsistema ya existía y no se ha
modificado su comportamiento).

## Arquitectura actual

El Skill System es un grafo inyectable definido en `core/skill_system.py`:

- `SkillRegistry` (core/skill_registry.py): registro determinista de
  `SkillDefinition` (id, nombre, versión, descripción, enabled,
  capacidades/permisos requeridos, tipos de agente permitidos, límites).
- `SkillManifestLoader` (core/skill_manifest.py): carga segura de manifests
  JSON (schema 1.0, máximo 64 KB, redacción de claves sensibles).
- `SkillDiscovery` (core/skill_discovery.py): descubrimiento de manifests en
  directorios con límites estrictos (128 directorios, 512 ficheros).
- `SkillRegistrationService` (core/skill_registration.py): discovery +
  loader + registro.
- `SkillResolver` (core/skill_resolver.py): resolución de skills requeridas.
- `SkillExecutor` (core/skill_executor.py): ejecución controlada con
  timeouts, cancelación cooperativa (`SkillExecutionContext`) y estados
  terminales estables.

Bootstrap: `bootstrap/skill_system.py` construye el sistema y registra los
skills builtin desde `skills/builtin/`. Cableado en `bootstrap/bootstrap.py`
y `core/agent_system.py`.

## Ciclo request → intent → resolver → executor

1. El texto del usuario entra como `AtlasRequest`.
2. `core/skill_intent.py::requested_skill_id` detecta intención explícita
   (el prompt debe contener "skill" y coincidir con id o nombre de una skill
   habilitada registrada). Si no hay keyword, no hay intención.
3. `core/skill_intent.py::skill_inputs_from_text` extrae inputs entrecomillados
   o etiquetados (`texto: ...`) solo para skills cuyo campo de entrada es
   exactamente `text`.
4. `SkillSystem.skill_resolver.resolve` resuelve la skill por id.
5. `SkillSystem.skill_executor.execute` ejecuta la skill con la política por
   defecto y metadatos de origen.
6. `core/skill_intent.py::present_skill_output` presenta el output de forma
   segura: prefiere la clave `result`, un único valor directo, o pares
   clave/valor.

Los planes multi-agente usan las mismas piezas vía `required_skill_ids`
(core/agent_cooperation_plan.py) con autorización por agente
(`allowed_agent_types`, `required_capability_ids`, `required_permission_ids`).

## Límites y permisos

- `SkillLimits`: timeouts y límites por skill.
- Autorización: una skill solo es ejecutable por agentes cuyos metadatos la
  permiten y que posean las capacidades y permisos requeridos.
- Skills deshabilitadas (`enabled=false`) nunca se resuelven ni ejecutan.

## Catálogo

`core/skill_catalog.py::build_skill_catalog(skill_system)` devuelve una vista
segura con allowlist explícita: únicamente `id`, `name`, `description`,
`version` y `enabled`. Nunca expone metadata, permisos internos, targets,
handlers ni configuración.

Política de visibilidad: coherente con `SkillRegistry.list_skills`; por
defecto lista todas las registradas marcando las deshabilitadas con
`enabled=False`. Con `enabled_only=True` se excluyen.

## Discovery y registration

Nuevos manifests se colocan bajo `skills/builtin/<skill_id>/skill.json` y se
registran al arrancar. El loader valida schema, tamaño y sanea claves
sensibles antes de crear la `SkillDefinition`.

## Cómo añadir un skill builtin nuevo

1. Crear `skills/builtin/<id>/skill.json` conforme al schema 1.0.
2. Declarar `input_fields`/`output_fields` y `execution_target`.
3. Registrar el handler en `build_builtin_skill_handler_registry()`
   si el target es de tipo handler.
4. Arrancar Atlas: el discovery/registration lo registra automáticamente.

## Skills propios (futuro)

Apuntar el discovery a directorios adicionales pasando `root_directories`
propios a `SkillDiscoveryRequest`, o construir un `SkillSystem` dedicado con
su propio registry mediante `build_skill_system(...)`. No existe todavía un
mecanismo de instalación dinámica en caliente.

## Límites de seguridad

- Manifests limitados a 64 KB; claves sensibles redactadas.
- Ejecución acotada por timeout y cancelación cooperativa; sin escapes.
- El catálogo y la presentación de salida son allowlist-based: sin paths,
  handlers, permisos, secretos ni metadata interna.
- Sin llamadas externas ni efectos secundarios fuera del executor.
