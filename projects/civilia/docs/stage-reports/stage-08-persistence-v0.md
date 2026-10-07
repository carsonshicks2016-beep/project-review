# Stage 8 — Persistence v0

**Status: complete (v0)** — 2026-06-11

## Delivered
- `src/sim/persistence.ts`: versioned JSON snapshot envelope (`SAVE_VERSION`), serialize/deserialize with shape validation
- Save/Load in the UI (localStorage slot per browser; wired in the WorldMap pass)
- Tests (`tests/sim/persistence.test.ts`):
  - lossless round-trip
  - **save mid-phase → load → continue replays the exact uninterrupted history** (works because randomness derives from `seed:day:phase:agent` labels, never from generator state)
  - rejects wrong versions and junk input

## Deferred (with reason)
- SQLite event-log tables, periodic snapshots, timeline branching: requires a Node server process, which the browser-only app doesn't have yet. Lands with the stage 29 Ollama gateway server. Ledger updated.
