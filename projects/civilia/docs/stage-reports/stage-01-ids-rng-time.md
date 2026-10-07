# Stage 1 — Core IDs, RNG, Time

**Status: complete** (audited 2026-06-11)

## Delivered
- Stable string IDs: `agent_0001`, `place_market`, `event_000001`, `rumor_0001`, `trace_000001`
- `src/sim/rng.ts`: xmur3 + mulberry32 seeded streams. Streams derive from labels (`seed:purpose:…`) so they are **order-independent** — adding a new consumer never reshuffles existing worlds. Trait rolls use one stream per trait for the same reason.
- Six day phases (dawn→night), `stepPhase`/`stepDay`, `world.day` + `phaseIndex`

## Verified
- `tests/sim/determinism.test.ts`: same seed ⇒ identical event log; different seed ⇒ diverges; traits stable per seed.

## Reinforcement notes
- No changes needed. The labeled-stream design is the project's strongest determinism guarantee; new systems (terrain, claims, speech) must derive streams the same way.
