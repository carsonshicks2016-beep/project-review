# Agent D — Three.js renderer

**Files:** `src/renderer/`, `src/main/`  ·  **Contract in:** `WorldSpec`,
`AudioFrame` (WS)

## Goal
Render a `WorldSpec` as a living 3D world inside the Electron window.

## Scope
- Three.js (WebGL2): scene, camera, lights, fog, bloom post-processing.
- Build terrain from `TerrainParams`; place `scatter` via GPU instancing.
- Mount the weather (Agent E) and creature (Agent F) systems.
- Connect to `AUDIO_WS_URL`; apply `modifierCurves` to AudioFrame each frame.
- Crossfade/dissolve between worlds on track change (~2–4s) — no hard cut.
- Run gracefully when the audio helper is offline (scenes self-animate).

## Acceptance
- Given a hand-written `WorldSpec`, renders one biome at ~60 fps.
- Live AudioFrame visibly drives brightness/particles; missing WS → still runs.
- Swapping the WorldSpec crossfades smoothly.
