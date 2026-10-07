# Stage 13 — Relationships (repair + favor economy)

**Status: complete** — 2026-06-11

## Delivered
- **Grievance tracking** on directed relationships (`grievances`, `lastHarmDay`): insults, witnessed thefts, and border confrontations now leave debts of harm that persist until repaired (lightweight precursor to stage 14 memory)
- **`apologize` action** with the spec's forgiveness thresholds — acceptance depends on sincerity (empathy), timing (fresh wounds forgive easier, stale ones insult twice), restitution (coins offered when the offender can spare them), the offended party's trust and pride, witnesses (public apologies flatter), and wound depth. Accepted: grievances cleared, resentment −0.22, trust up, both parties' belonging up. Rebuffed: status loss (worse in public) and fresh bitterness — so proud agents rarely risk it
- **Favor economy v0**: cross-household gifts create favor debts (family shares freely); `repay_favor` clears them and warms the creditor; `applyDebtPressure` makes unpaid debts corrode the creditor's patience nightly
- Speech + motives for both: apologies classify as genuine / performative (public + proud) / premeditated (fence-mending upward in status) / desperate (too lonely to afford an enemy); repayments as pride or bonding
- **Social graph UI** (People tab): villagers in household-colored circle, edges for kinship, affection (green), resentment (red), open debts (gold dashed), hover for both directions' numbers + grievances, click to open the agent drawer
- `SAVE_VERSION` bumped to 2 (relationship shape changed)

## Verified
- `tests/sim/relationships.test.ts` (7): favor debts (and the family exemption), repayment effects, nightly debt pressure, grievance recording, engineered accept/reject apologies hitting the thresholds, repairs occurring in real runs with deterministic replay. Suite 61/61, tsc clean.
- 40-day probe (default seed): 36 apologies (23 accepted), 2 repayments, 20 open debts. People tab verified in-browser; zero console errors.

## Notes
- The unemployed accumulate debts they can't repay — emergent class texture, exactly what the spec's "Trust-Based Credit" wants. Stage 18 (economy pressure) can build on this.
