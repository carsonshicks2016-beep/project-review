# Agent C — Generation core (pure)

**Files:** `src/generation/`  ·  **Contract in:** `TrackContext` (+ optional
`AudioFrame` prior)  ·  **Contract out:** `WorldSpec`

## Goal
Deterministically turn a track into a fully-described world.

## Scope
- `seedFromTrackId(trackId)` — stable string hash (e.g. cyrb53/xfnv1a).
- Seeded PRNG (`seedrandom`) drives terrain (`simplex-noise` heightfield),
  Poisson-disk `scatter`, `creaturePool` species selection, `timeOfDay` jitter.
- Genre → `Biome` registry per `PLAN.md §4`; **mood fallback** from the audio
  prior (energy + brightness) when `genres` is empty/ambiguous.
- Map `palette` onto the world; set `weatherState` and `modifierCurves`.

## Hard constraints
- **Pure & deterministic:** same `trackId` ⇒ identical `WorldSpec`. No I/O,
  no `Date.now()`, no `Math.random()`.

## Acceptance
- Unit tests: same id → deep-equal WorldSpec; different ids → different worlds.
- Every `Biome` is reachable; unknown genre + audio prior resolves sensibly.
