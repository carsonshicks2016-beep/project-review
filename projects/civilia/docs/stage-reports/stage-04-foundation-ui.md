# Stage 4 — Foundation UI

**Status: complete, superseded in part** (audited 2026-06-11)

## Delivered
- Map view, searchable event timeline, stats bar, event ticker, agent drawer with needs/traits/relationships/rumors, place cards, seed box, play/pause/speed, step phase/day
- Chronicle reader + config (built early, out of order — deterministic side of stage 28)

## Reinforcement notes (this pass)
- The hand-placed SVG town map (`TownMap.tsx`, fixed `PLACE_POS`) is replaced by the procedural `WorldMap` (stages 9–12): canvas terrain, three zoom tiers, 16-bit style. Timeline, drawers, ticker, stats, and chronicle panels are retained.
