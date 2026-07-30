# Preguntas para revisores externos

## Arquitectura

1. Es solida la separacion actual entre entrada, routing, planning, gobierno,
   ejecucion, supervision y verificacion?
2. Que responsabilidades estan mal ubicadas en AtlasOrchestrator,
   StructuredExecutionCoordinator o Bootstrap?
3. Hay duplicaciones reales o solo fachadas legitimas sobre un nucleo comun?
4. Debe V2 elegir una sola entrada entre `process_prompt`,
   `process_prompt_result` y `route_structured_input`?
5. Que contratos deben declararse estables antes de modularizar?

## Escalabilidad y mantenibilidad

6. Cuales son los primeros limites de los modulos de 1000-5700 lineas?
7. Como dividirlos sin reconstruir el motor ni perder invariantes?
8. La composicion manual en Bootstrap es sostenible?
9. Que observabilidad falta para diagnosticar produccion local?
10. Que deuda debe resolverse antes de anadir nuevas capacidades?

## Seguridad

11. Son suficientes firma, autorizacion y Dispatcher para el modelo local?
12. Debe persistirse el ledger de dispatch y confirmaciones consumidas?
13. Que herramientas de escritorio necesitan confirmacion obligatoria?
14. Debe cifrarse persistencia/logs o bastan permisos locales?
15. Que estrategia recomiendan para retirar `.env` del historial?
16. Donde puede filtrarse contenido privado pese a la sanitizacion actual?

## Multiagente, skills y capabilities

17. Es correcta la convivencia entre tres agentes clasicos y AgentSystem
    declarativo vacio?
18. Que parte del sistema multiagente esta preparada y cual es solo
    infraestructura?
19. AgentExecutor, SkillExecutor y CapabilityOrchestrator reutilizan
    adecuadamente el motor o crean gobernanza paralela?
20. Que invariantes deben imponerse antes de activar agentes declarativos?
21. Conviene consolidar registries o mantener adapters explicitos?

## Preparacion V2

22. Que cambios son prioritarios para Windows sin ampliar privilegios?
23. Como aislar voz para que no penalice el modo texto?
24. Donde debe vivir la seleccion de LLM y su politica de timeout/fallback?
25. Que limites de seguridad necesita una futura integracion WhatsApp?
26. Que API deberia consumir una futura interfaz grafica?
27. Que elementos de v1.0 no conviene modificar por su estabilidad demostrada?
28. Que tres mejoras aportan mayor valor con menor riesgo?

## Forma esperada de respuesta

Para cada hallazgo se solicita:

- severidad;
- evidencia concreta en archivo/contrato;
- impacto;
- recomendacion incremental;
- alternativa;
- riesgo de migracion;
- confianza del revisor.
