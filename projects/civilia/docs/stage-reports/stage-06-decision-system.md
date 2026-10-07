# Stage 6 — Decision System

**Status: complete** (audited 2026-06-11)

## Delivered
- `decisionSystem.ts`: gathers candidates from every action, adds seeded noise (±7), selects by squared-score weighted randomness from the top-5 pool — motivated but not perfect choices, per the spec's "free will" model
- Idle is always a candidate (wins when nothing presses)
- Full `DecisionTrace` per agent per phase: every candidate, score, noise, reasons, chosen index, emitted event ids (capped ring of 30 per agent)

## Reinforcement notes
- The trace is the substrate for stage 24's phrase-motive inspector: "why did they say that" reduces to "why did they do that" plus relationship context.
