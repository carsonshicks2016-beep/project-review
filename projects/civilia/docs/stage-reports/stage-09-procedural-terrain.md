# Stage 9 — Procedural Terrain v0

**Status: complete** — 2026-06-11

## Delivered
- `src/sim/terrain.ts`: 96×64 seeded landscape as canonical sim state
  - Four-octave value noise (cached lattice from labeled rng streams — order-independent, replay-safe)
  - Western ocean shelf, eastward rise; edge-connected water = ocean, the rest lakes (one inland lake guaranteed — the town needs fresh water to find)
  - Rivers descend from eastern springs, carving so replays never flow uphill
  - 13 biomes: ocean, lake, river, beach, marsh, clay_flats, grass, meadow, forest, dense_forest, hills, mountain, snow
  - Gatherable materials per tile: timber (forests), stone (mountains/hills), clay (riverbanks), reeds (marsh) — stage 19's construction feeds on these
  - **Named landmarks**: the sea, largest lake, highest peak, largest forest get seeded names ("Mirror Tarn", "Raven Fell") — geography that speech and rumors can reference
  - Deterministic town-site scan: open buildable neighborhood + fresh water ≤ 8 tiles + timber in reach
- `createWorld`: places settle onto tiles around the town center (`TOWN_LAYOUT` offsets + ring-search `findBuildSite`), lots cleared, farm gets a field
- `World.terrain`, `World.pathWear` (stage 12 substrate), `Place.tile`

## Verified
- `tests/sim/terrain.test.ts` (6): determinism per seed, biome variety (ocean/freshwater/forest/highland), materials present, landmarks named, town site interior + walkable, every place on a unique buildable tile. Full suite 36/36.
