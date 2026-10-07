# Legacy 3D Viewer (v1)

The original Three.js viewer ("mountain pass", CRT-retro look: 0.56 render
scale, dither quantize, scanlines), frozen 2026-06-09 when the from-scratch v2
rebuild started (see `VIEWER3D_V2_PLAN.md` at the repo root).

- Served at **`/3d/legacy/`** (v2 lives at `/3d/`). Shares the same
  `/api/3d/ws` physics bridge.
- This is the exact pre-v2 working tree; the only change is the two asset URLs
  in `index.html` (rewritten `/3d/static/…` → `/3d/legacy/static/…`).
- `viewer3d/graphics_v1/` is an older approved snapshot of this viewer
  (pre day/night toggle); kept as an archive.
- **Do not edit this directory** — it exists as a rollback/reference.
