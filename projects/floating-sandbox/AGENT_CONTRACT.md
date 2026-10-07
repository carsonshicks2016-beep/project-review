# Floating Sandbox — Agent Contract

Build a browser 2D Floating Sandbox remake. Target: **100,000+ water particles** via WebGPU when available; CPU fallback must work (~8–15k).

## Repo layout (create only your assigned files)

```
src/
  sim/types.ts          ✅ EXISTS — do not change unless necessary
  sim/materials.ts      ✅ EXISTS
  sim/water/            Agent A
  sim/ship/             Agent B
  sim/coupling/         Agent C
  sim/vessels/          Agent D
  sim/tools/            Agent E
  render/               Agent F
  ui/                   Agent G
  app.ts / main.ts      Integrator (parent)
```

## Physics priorities (correctness > flash)

1. Fixed timestep + substeps
2. PBF water (spacing, density/pressure, viscosity, mild surface tension, gravity)
3. XPBD mass-spring ships with tension/compression break + ductility
4. Watertight hull skin; breaches flood compartments; flooded mass weighs down
5. Particle↔hull collision + buoyancy + drag
6. Bombs = radial impulses (tear by strain), not delete-circle

## Visual style

Clean simulation UI — NOT glassmorphism, neon, purple gradients, emoji, glow spam.
Soft sky + seabed OK. Readable silhouettes, metaball-ish water, optional stress overlay.

## Non-goals for v1

No fire/steam/audio polish. First: float → smash → flood → sink. Then bombs/cuts. Then blueprint PNG.

## Exports each agent must provide

See your task prompt for exact export signatures. Import types from `../types` or `../../sim/types`.
