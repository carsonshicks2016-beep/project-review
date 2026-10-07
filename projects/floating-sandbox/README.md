# Floating Sandbox

2D ship-in-water physics toy — drop vessels, smash holes, watch them flood and sink.

## Run

```bash
npm install
npm run dev
```

Open http://localhost:5173/

## Controls

| Input | Action |
|-------|--------|
| 1–8 | Tools (drag, smash, cut, bomb, pin, repair, add/remove water) |
| Left drag | Use active tool |
| Alt + drag / middle mouse | Pan |
| Wheel | Zoom |
| Panel | Spawn vessels, density/strength, particle count, GPU, pause/step/reset, PNG blueprint |

## Architecture

- **CPU PBF** water (default) — correct coupling with ships
- **WebGPU PBF** — enable in panel for large oceans (up to ~131k capacity); readback each step for CPU coupling
- **XPBD** mass-spring ships with breakable beams, ductility, braces
- **Flooding** from broken hull skin → compartment flood mass → sink
- Fixed timestep sim loop in `src/sim/world.ts`

## Milestone checklist

1. Drop liner → smash below waterline → flood → sink
2. Bombs/cuts tear by strain
3. Blueprint PNG import (opaque pixels → nodes; colors → materials)

## Notes

- Default ~5k particles on CPU for interactive rates. Raise the Particles slider and tick **GPU (WebGPU)** for 50k–100k+.
- Correctness over flash: start CPU, then scale.
