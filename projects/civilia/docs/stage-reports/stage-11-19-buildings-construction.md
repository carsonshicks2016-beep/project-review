# Stage 11 — Buildings/Occupancy + Stage 19 v0 — Construction System

**Status: stage 11 complete; stage 19 v0 complete (deepening reserved for its slot)** — 2026-06-11

## Delivered
- `Building` model: kind, state (**planned / under_construction / complete**), progress, material recipe, owner household, proposal day + reason — construction is always motivated
- Founding town stands pre-built; workplaces owned by the households that run them (farm→Calder, bakery→Bray, tavern→Hale), market/well civic
- **Need-driven proposals** (`constructionSystem.ts`, end of day):
  - ≥2 bread-shortage price events in a week and no granary → the town plans a granary by the bakery
  - A household with 3+ members and ≥20 pooled coins (after day 5) → a cottage for the youngest adult
- **Materials from the land**: builders gather timber/stone/clay/reeds from the nearest terrain tiles, depleting them; recipes **adapt to local availability** (scarce materials substitute toward timber) so projects can't stall on geology
- `build` action for laborers and the unemployed: gather trips, then build shifts (+levy wage), completion at full progress; building sites call to free hands via travel scoring
- **Completion changes the town**: granary buffers farm grain and extends the baker's batch; a finished cottage spins off a **new household** (`household_formed`) — which automatically becomes a Chronicle faction
- Indoor/outdoor occupancy (`occupancySystem.ts`): sleeping/indoor work/evening home time put agents inside; fields, stalls, and travel keep them out — feeds the map's "who's inside" view

## Verified
- `tests/sim/construction.test.ts` (6): founding buildings + ownership, cottage proposal triggers, gathering depletes terrain and completes adapted recipes, full 40-day run produces completed buildings and a spun-off household, 20-day determinism, occupancy tracked. Suite 49/49, tsc clean.

## Deferred to stage 19/20 proper
- More building types (workshop, shrine, walls), repair/decay, material quality affecting status, construction site disputes.
