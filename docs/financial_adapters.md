# Hito: adaptadores financieros

**Estado:** decisión provisional de arquitectura; documentación únicamente.

> **Advertencia visible:** Atlas no conecta cuentas, no envía órdenes y no usa
> red financiera ni credenciales como parte de este hito. La información de este
> documento no demuestra rentabilidad ni constituye una recomendación de
> inversión.

## Estado de la investigación V2

- **Decisión:** **PAUSADA**. No se lanzarán nuevas capturas ni se conectarán
  brokers.
- La revisión de los ledgers existentes encontró como máximo **4 eventos
  independientes utilizables por pareja de captura y activo**.
- El mínimo **30** citado en los análisis es un umbral exploratorio del código,
  no una garantía de suficiencia estadística.
- Los costes modelados no son tarifas reales verificadas.
- Existen problemas de atribución en ledgers legacy y algún horizonte incompleto.
- La evidencia actual no permite afirmar rentabilidad ni elegir parámetros.
- V2 permanece desconectada de `ExecutionIntent` y `REAL_EXECUTION_DISABLED`.
- Para reabrir la investigación hace falta una hipótesis revisada y un
  protocolo de evaluación definido antes de recoger más datos.

## Decisión provisional

- **IBKR** será el adaptador previsto para acciones y ETF.
- **Kraken** será el adaptador previsto para cripto spot.
- Los adaptadores serán componentes separados. No se implementan aquí, no se
  conectan cuentas y no se conecta V2 con `ExecutionIntent`.

La TWS API de IBKR es una API de socket que se conecta a TWS o IB Gateway y
permite intercambiar mensajes con la plataforma de IBKR: [TWS API oficial](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/). La
selección de tipos de orden debe contrastarse con la tabla oficial de [IBKR
Order Types](https://www.interactivebrokers.com/campus/ibkr-api-page/order-types/).

## Alcance inicial

Inicialmente solo se permitirían:

- acciones y ETF admitidos por IBKR, sujeto a disponibilidad, permisos y reglas
  del instrumento;
- pares de cripto spot admitidos por Kraken, sujeto a disponibilidad, precisión,
  mínimos y reglas de la pareja. La documentación de [Kraken Add
  Order](https://docs.kraken.com/api-reference/trading/add-order) remite a
  `AssetPairs` para esos parámetros.

Quedan excluidos explícitamente del alcance inicial:

- margen y apalancamiento;
- derivados, futuros, opciones y productos equivalentes;
- retiradas de fondos.

Los tipos de orden iniciales serían únicamente **MARKET** y **LIMIT**, siempre
sujetos a validación por instrumento o pareja, precisión, mínimos, saldo,
permisos y reglas del mercado. En Kraken, la fuente oficial enumera `market` y
`limit` entre los tipos de `AddOrder`; no se habilitarían por ello los demás
tipos descritos allí.

## Límites operativos registrados

- **IBKR:** se registra como límite de diseño inicial un máximo de **50
  mensajes por segundo** y hasta **20 órdenes abiertas por contrato, lado y
  cuenta**. La cifra exacta y la semántica de la agrupación quedan **pendientes
  de verificación en la documentación oficial**; la referencia inicial es la
  [TWS API oficial](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/).
  No se debe implementar un controlador basándose únicamente en esta nota hasta
  completar esa verificación.
- **Kraken:** los límites son variables según el nivel de la cuenta y la pareja,
  y existen límites diferenciados del contador REST y del motor de matching.
  La [guía oficial de límites REST spot](https://docs.kraken.com/api/docs/guides/spot-rest-ratelimits/)
  documenta niveles Starter, Intermediate y Pro, sus contadores y tasas de
  decaimiento, y remite además a límites del motor de trading. La cifra efectiva
  deberá resolverse por cuenta, endpoint y pareja antes de operar.

## Permisos mínimos futuros

La configuración futura debe aplicar el principio de mínimo privilegio:

- **IBKR:** solo permisos de cuenta/API necesarios para consultar datos y
  gestionar órdenes de los instrumentos autorizados; los nombres exactos y la
  configuración mínima quedan pendientes de verificación por tipo de cuenta y
  canal (TWS o IB Gateway).
- **Kraken:** permiso para crear/modificar órdenes y, cuando sea necesario,
  permiso para cancelar/cerrar órdenes. [Add Order oficial](https://docs.kraken.com/api-reference/trading/add-order) requiere `Orders
  and trades - Create & modify orders`; [Cancel Order oficial](https://docs.kraken.com/api-reference/trading/cancel-order) admite ese
  permiso o `Orders and trades - Cancel & close orders`.
- En Kraken se debe **excluir siempre el permiso de retirada**. No se solicitará,
  almacenará ni habilitará una credencial con capacidad de retirar fondos.

## Seguimiento y conciliación

Cada orden futura deberá conservar, como mínimo:

- un identificador de intención interno y un identificador de cliente
  idempotente;
- broker, cuenta, instrumento o pareja, lado, tipo, cantidad, precio y
  parámetros normalizados;
- identificadores del broker, estado, timestamps, cantidades ejecutadas,
  precio medio, comisiones y mensajes de rechazo;
- relación entre solicitudes, actualizaciones, cancelaciones y ejecuciones.

El adaptador deberá poder consultar el estado, recibir o recuperar
actualizaciones, solicitar cancelación y distinguir una cancelación aceptada de
una cancelación pendiente. En Kraken, `AddOrder` devuelve `txid` y acepta
`cl_ord_id`; `CancelOrder` permite cancelar por `txid`, `userref` o `cl_ord_id` y
puede indicar que la cancelación está pendiente. Esas afirmaciones están
respaldadas por [Add Order](https://docs.kraken.com/api-reference/trading/add-order)
y [Cancel Order](https://docs.kraken.com/api-reference/trading/cancel-order).

La conciliación deberá ser periódica y también dispararse tras errores de red,
reinicios, timeouts o respuestas ambiguas. No se asumirá que un timeout implica
rechazo: se consultará el estado remoto, se reconciliarán ejecuciones y se
registrará cualquier divergencia para detener nuevas acciones. Las reglas
concretas de consulta y eventos de IBKR quedan pendientes de verificación en la
documentación oficial de la TWS API.

## Requisitos de Atlas

Antes de cualquier implementación futura, el diseño deberá conservar estas
garantías:

- adaptadores separados para IBKR y Kraken, con contratos explícitos y sin
  compartir transporte ni credenciales;
- `RiskGate` independiente del adaptador y capaz de rechazar antes de enviar;
- autorización humana explícita para cada operación autorizable;
- fallo cerrado: error, timeout, estado desconocido o divergencia bloquea nuevas
  operaciones hasta conciliación y decisión segura;
- idempotencia en solicitudes, reintentos controlados y cancelaciones;
- `REAL_EXECUTION_DISABLED` como bloqueo efectivo por defecto y durante toda
  esta fase documental.

Las pruebas futuras se limitarán a transportes falsos, respuestas controladas y
datos sintéticos. No se usarán cuentas reales, credenciales, red financiera ni
órdenes reales.

## Fuentes oficiales iniciales

- [IBKR TWS API](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/)
- [IBKR Order Types](https://www.interactivebrokers.com/campus/ibkr-api-page/order-types/)
- [Kraken Add Order](https://docs.kraken.com/api-reference/trading/add-order)
- [Kraken Cancel Order](https://docs.kraken.com/api-reference/trading/cancel-order)
- [Kraken Spot REST rate limits](https://docs.kraken.com/api/docs/guides/spot-rest-ratelimits/)

Las afirmaciones de límites concretos de IBKR y de permisos mínimos exactos por
tipo de cuenta quedan pendientes de verificación oficial antes de cualquier
conexión o implementación.
