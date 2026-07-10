# SPEC-001 — Atlas Core Foundation

## Estado

Draft

---

## Objetivo

Definir el núcleo (Core) de Atlas.

El Core contendrá únicamente los componentes fundamentales compartidos por todo el sistema.

No tendrá dependencias hacia otros módulos.

Todos los demás módulos dependerán del Core.

---

## Responsabilidades

El Core será responsable de proporcionar:

- Tipos base
- Interfaces comunes
- Objetos de dominio
- Errores comunes
- Utilidades compartidas

---

## Restricciones

El Core NO podrá depender de:

- Providers
- Models
- Agents
- Memory
- Planner
- Tool System
- Router
- Orchestrator

---

## Componentes iniciales

- Result
- Error
- Identifier
- ValueObject
- Entity
- AggregateRoot
- Command
- Query
- DomainEvent
- Clock

---

## Criterios de aceptación

- Sin dependencias externas innecesarias.
- Código desacoplado.
- Interfaces claras.
- Compatible con Clean Architecture.
- Compatible con SOLID.

---

## Estado

Pendiente de implementación.