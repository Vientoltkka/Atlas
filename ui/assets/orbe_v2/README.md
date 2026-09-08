# orbe_v2 — Assets del Orbe ATLAS (capa gráfica por composición)

Assets PNG con canal alpha real, generados programáticamente por
`generate_assets.py` (PySide6/Qt, sin dependencias externas). El runtime los
compone por capas en `ui/orb_assets.py` + `ui/orbe_app.py` (`_draw_asset_device`).

## Asset hero APROBADO (composición única)

`atlas_orbe_approved.png` (1295×1214, RGBA con alpha real) es el asset visual
aprobado por el usuario: esfera + logo + glow + órbitas + partículas + pedestal
en una sola imagen. Cuando está en disco, `paintEvent` lo dibuja como el ÚNICO
visual del Orbe (`_draw_hero_device`, sin rotación, solo respiración/pulso y
tinte por estado) y NO superpone las capas provisionales ni las órbitas
procedurales. Ventana: IDLE 560 px → esfera ≈ 439 px; estados activos 590-600 px.
Las capas por composición de abajo son SOLO fallback (si el hero falta).

## Orden de composición (paintEvent, fallback sin hero)

1. `globe_body.png` — esfera holográfica base (sombreado 3D, borde). Escala 0.30·d.
2. `globe_grid.png` — red de puntos/meridianos; rota lento (offset por tiempo). Escala 0.55·d.
3. `inner_glow.png` — halo interno. Escala 0.45·d.
4. `atlas_logo.png` — chevron Atlas con pulso. Escala 0.22·d.
5. Órbitas — **reutilizadas del renderer procedural existente** (sin asset).
6. `particles.png` — motas rotantes. Escala 0.55·d.

## Tintes por estado

Aplicados vía `CompositionMode_SourceAtop` (conserva alpha y sombreado) en
`orb_assets.state_bundle(state)`: SPEAKING (verde), AUTHORIZATION, AUTOMATION
(núcleo oscuro + tinte rojo), DEGRADED. IDLE/PROCESSING/LISTENING comparten el
asset neutro cian (mismo objeto en caché).

## Regeneración

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python ui/assets/orbe_v2/generate_assets.py
```

## Assets definitivos aún pendientes (veredicto B)

La infraestructura de carga/composición/tinte está completa y probada; estos
assets sustituibles elevarían la fidelidad sin tocar código (mismo nombre,
mismo tamaño base, RGBA):

| Asset | Estado actual | Reemplazo definitivo sugerido |
|---|---|---|
| `globe_body.png` (1024²) | Gradiente radial procedural | Render 3D profesional (Blender/CDN autorizado) con terminador suave y fresnel en el borde |
| `globe_grid.png` (1024²) | Puntos por latitud + nodos + meridianos | Textura de red/constelación de fidelidad superior (wireframe HR) |
| `atlas_logo.png` (512²) | Chevron vectorial con glow | Logo Atlas oficial renderizado con luz volumétrica |
| `inner_glow.png` (512²) | Halo radial simple | Halo con caústicas/ruido volumétrico |
| `particles.png` (512²) | Motas dispersas procedurales | Hoja de partículas con variación de brillo/tamaño (bokeh) |

Variante dedicada de núcleo rojo para AUTOMATION (opcional): hoy se logra
oscureciendo el cuerpo + tinte; un `globe_body_automation.png` dedicado daría
más contraste.

## Fallback

Si falta cualquier PNG en disco, `paintEvent` cae automáticamente al renderer
procedural `_draw_device`/`_draw_core_particles`/`_draw_emblem` (nunca crashea).
