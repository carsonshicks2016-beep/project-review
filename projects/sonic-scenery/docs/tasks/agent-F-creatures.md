# Agent F — Creature system

**Files:** `src/creatures/`  ·  **Contract in:** `WorldSpec`, `AudioFrame`
·  **Depends on:** Agent D scene conventions

## Goal
Populate the world with ambient creatures matching the biome.

## Scope
- Behaviors: Reynolds `boids` (birds/fish/fireflies), `wander` (ground),
  `drift` (jellyfish/dream).
- GPU instancing; per-species from `CreatureSpec` (`count`, `baseSpeed`).
- Onset reactions: bursts/scatter/speed-ups from live `AudioFrame`.

## Acceptance
- A flock obeys cohesion/separation/alignment and stays in bounds.
- Counts/speeds derive from `CreatureSpec`; onsets cause visible reactions;
  offline audio → calm ambient motion. Holds 60 fps at target counts.
