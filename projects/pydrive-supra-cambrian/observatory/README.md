# Fable Observatory

A clean-room Three.js browser viewer for authoritative Fable Five playback on the runtime-smoothed Nordschleife. It supports the project-authored Mazda 787B and Porsche 919 Evo visualization packages while leaving the existing 2D viewer unchanged.

## Build

```bash
cd observatory
npm install
npm run assets
npm test
npm run build
```

`npm run assets:check` regenerates every binary in memory and byte-compares it with the checked-in artifact. The world manifest records the runtime track hash and source hashes. Road and shoulder truth use all 6,944 simulator samples. Terrain comes from the 21 required cached `.cache/nordschleife_dgm1/*.tif` DGM1 GeoTIFFs as 25 m and 100 m LOD tiles; the builder never downloads them and fails if an input is missing or its hash drifts. Terrain, curbs, trees, fencing, guardrails, and landmark-inspired markers are explicitly visual-only and non-colliding.

## Launch contract

The dashboard normally opens the viewer with `edition`, `checkpoint`, and `mode` query parameters; the server resolves the authoritative car identity. The client creates a browser-owned session through `POST /api/observatory/sessions`, then uses the returned telemetry and audio socket paths. Their standard routes are `/api/observatory/ws/<session-id>` and `/api/observatory/audio/<session-id>`. Engine audio connects only after a user gesture.

For an already-created session, use `?session=<session-id>`. Broadcast is the default camera; chase, roof, trackside, drone-orbit, and free cameras are also available. Dry, dusk, and night lighting do not mutate simulation state.

Every view keeps checkpoint filename, policy SHA-256, vehicle identity, and compatibility classification visible. The Brain X-Ray, Engineer, and Replay panels are display and transport surfaces only: the Python playback service remains authoritative for physics, observations, policy inference, and scrubbing.
