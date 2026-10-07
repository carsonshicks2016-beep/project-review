# Stage 15 — Belief System + Dynamic Contextual Dialogue + Rumor Lens

**Status: complete** — 2026-06-11 (implements the Living World Architecture's dialogue engine + rumor overlay; fog/dollhouse/materials awaiting visual direction)

## Belief system
- `Belief` model: graded confidence (never binary), source (`witnessed` / `rumor` / `inference`), canonical `truthStatus` for the truth-vs-belief inspector, emotional charge, heard-from chain
- **Eyewitnesses know** (0.95, gossip barely moves them — receptivity 0.12); **listeners weigh the teller**: source trust, prior feeling about the subject (grudges make accusations easy, affection makes them hard), hunger + real shortages corroborate bread stories, own eyewitness history corroborates accusations (+25)
- **Belief inertia**: new tellings move confidence partway (0.45); repetition has diminishing returns
- Rumor believer/skeptic rosters now derived from confidence thresholds (≥0.6 / ≤0.35), with an "unsure" middle
- Founding suspicion: the bakery-secret knower starts with a 55% inference, canonically true
- Damaging convictions bias insults with visible reasons; belief inspector in the agent drawer (source icon, ✓/✗/? canonical-truth mark, confidence bar, heard-from)

## Dynamic Contextual Dialogue Engine
- Insults/help query the speaker's memory graph for the strongest edge to the listener and speak about *that*: "I saw you with my own eyes this very day… Once a thief, always a thief" / "After what you did for me 7 days back, this is the least of it"
- Fallback chain: wound memory → damaging belief ("Everyone whispers it, so I'll say it plain: …") → stock lines
- **Citations are speaking-time snapshots** on the utterance — the memory may fade, the receipt survives (the failing-test discovery that shaped the design)
- Reasoning Tree in the PhraseCard: Motivation → Historical context (cited memory, day formed, days elapsed, dwell count, origin event + phase, "since faded" marker) → Influencing knowledge (belief, confidence then vs now, source, who told them) → "How it landed" (gossip reception factors)
- 20-day probe: 123 of 482 spoken lines cite a specific memory or belief

## Rumor River overlay (v0)
- HUD lens (🗣): recent tellings as glowing arcs with traveling particles between teller and listener — purple for faithful retellings, **pink for mutations**, particle size/speed by emotional charge; terrain desaturates under the lens

## Verified
- `tests/sim/beliefs.test.ts` (7): founding suspicion, witness-vs-listener asymmetry, disposition bias, inertia + diminishing repetition, roster syncing, citations in real runs, determinism. Suite 73/73, tsc clean. In-browser: reasoning tree traced a spoken grudge to its day-8 origin event; rumor lens verified visually. `SAVE_VERSION` 4.
