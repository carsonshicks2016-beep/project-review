# Stage 12 — Movement / Path Wear

**Status: complete** — 2026-06-11

## Delivered
- `src/sim/movement.ts`: A* tile routing between places — ocean/lake impassable, rivers fordable but slow, forest/hills/marsh costlier, deterministic tie-breaking
- **Path wear feedback loop**: every walked tile gains wear; worn tiles are up to ~1.8× cheaper, so popular routes self-reinforce trail → path → road (`WEAR_TRAIL`/`WEAR_PATH`/`WEAR_ROAD` thresholds shared with the renderer); nightly decay fades unused routes over weeks
- Travel action now walks real routes: fatigue scales with distance, `traveled` events carry distance
- Deliberately **no route caching**: wear changes mid-phase, and module-level cache state would break save/load replay identity (verified by persistence tests)

## Verified
- `tests/sim/movement.test.ts` (4): routes walkable + contiguous, wear accumulates and decays, worn routes stay preferred (≥60% overlap after heavy wear), full-run wear deterministic. Suite 40/40.
