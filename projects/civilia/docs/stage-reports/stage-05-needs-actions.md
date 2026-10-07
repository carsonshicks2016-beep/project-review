# Stage 5 — Basic Needs/Actions

**Status: complete** (audited 2026-06-11)

## Delivered
- Needs: hunger, fatigue, belonging, with phase drift + dawn reset (`needsSystem.ts`)
- Money as pressure (wages, prices, theft motive)
- Actions: work_shift (wages, bread/grain production), buy_food (shortage pricing), eat_food, sleep, travel_to_place — plus social/crime actions from later stages built early
- Each action: preconditions → scored candidates with reasons → execute → typed events

## Verified
- `tests/sim/actions.test.ts`: hunger drives eating, work pays wages, buying respects stock/money, theft emits witnessed events, gossip spreads rumors.

## Reinforcement notes
- Emotions (anger, fear, shame…) deliberately deferred to stage 14 with memory — the spec models them as event-driven spikes, which needs memory hooks.
