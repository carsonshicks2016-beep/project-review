# Stage 21 v0 — Claims/Borders (pulled forward for the living map)

**Status: v0 complete** — 2026-06-11

## Delivered
- `claimsSystem.ts`: households project territory around buildings they own (strength = radius − distance; radius grows with members, status, holdings; **rising buildings already project a weaker claim** — construction is politics)
- Near-equal projections mark tiles **disputed**; water can't be claimed
- Claims recompute every dawn from canonical building ownership (derived state), rendered as the region-zoom border fill
- **Border incidents**: disputed ground flares (seeded, ~25%/day) into confrontations between household heads — mutual resentment, a public `border_dispute` event with a spoken line, and a 50% chance of a boundary-stones rumor

## Verified
- `tests/sim/claims.test.ts` (5): homes sit in their own claim, water unclaimed, disputes require two claimants, 15-day determinism, incidents carry speech. 30-day probe: 57 disputed tiles, 6 incidents. Suite 54/54.

## Deferred to stage 21 proper
- Non-household factions (cults/clubs/governments — stage 37 provides them), claim treaties, boundary-stone artifacts (stage 25 hooks).
