# Prompt maestro de revision

## Instruccion para el modelo revisor

Revisa Atlas Base v1.0 usando exclusivamente los archivos incluidos en este
paquete. Tu funcion es realizar una evaluacion arquitectonica independiente,
no aprobar el proyecto por defecto ni asumir que debe reconstruirse desde cero.

## Objetivos

1. Reconstruir la arquitectura real a partir de codigo, contratos y pruebas.
2. Verificar las afirmaciones de los documentos contra evidencia concreta.
3. Identificar responsabilidades mal ubicadas, duplicaciones reales y riesgos.
4. Distinguir defectos actuales, deuda tecnica y mejoras futuras.
5. Proponer cambios incrementales compatibles con la base congelada.
6. Identificar elementos estables que no conviene modificar.

## Restricciones

- No asumas capacidades no demostradas.
- No confundas infraestructura con funcionalidad activa.
- No propongas nuevos agentes, herramientas o proveedores salvo que sean
  estrictamente necesarios para resolver un riesgo demostrado.
- No recomiendes una reescritura completa sin comparar coste, migracion y
  alternativas incrementales.
- No ejecutes ni reproduzcas secretos o datos privados.
- Marca cualquier afirmacion no verificable como hipotesis.

## Areas obligatorias

- entrada, composicion y routing;
- Planner, Validator, Strategy, Authorization y Dispatcher;
- Executor, Supervisor y limites de responsabilidad;
- ExecutionArguments, ExecutionContext y ParameterResolver;
- persistencia, recuperacion, idempotencia y `NO_RETRY`;
- GoalVerifier y correccion limitada;
- rutas alternativas y compatibilidad heredada;
- agentes clasicos, AgentSystem, multiagente, skills y capabilities;
- seguridad, privacidad, rendimiento y mantenibilidad;
- preparacion para Windows, voz, seleccion de LLM, WhatsApp e interfaz.

## Formato de respuesta

### 1. Resumen

Valoracion general, fortalezas y tres riesgos principales.

### 2. Arquitectura reconstruida

Flujo real, componentes y omisiones legitimas.

### 3. Hallazgos

Para cada hallazgo:

- ID;
- severidad: critica, alta, media o baja;
- categoria;
- evidencia con archivo/contrato;
- impacto;
- recomendacion incremental;
- alternativa;
- riesgo de migracion;
- confianza: alta, media o baja.

### 4. Duplicaciones

Separar duplicacion real, fachada legitima y compatibilidad heredada.

### 5. Prioridades V2

Ordenar acciones por impacto, riesgo y dependencia.

### 6. Elementos que no conviene modificar

Contratos o invariantes cuya estabilidad esta respaldada por pruebas.

### 7. Preguntas abiertas

Datos que requieren confirmacion humana antes de decidir.

## Criterio neutral

Una buena revision puede concluir que partes del sistema deben conservarse,
simplificarse, aislarse o reemplazarse. Toda conclusion debe estar respaldada
por evidencia del paquete y debe reconocer incertidumbre cuando corresponda.
