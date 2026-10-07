# Stages 23/24 v0 — Speech Bubbles & Phrase Inspection (sim side)

**Status: v0 complete (pulled forward — the living-map spec requires them)** — 2026-06-11

## Delivered
- `speechSystem.ts`: every social interaction (talk, gossip, insult, help, caught theft, construction chatter, border disputes) composes an actual spoken line from **preset templates × live context**: names, current bread price, the speaker's juiciest rumor, named landmarks, building kinds, relationship state
- Lines are stored on the emitting event (`payload.speech`) at emit time — canonical, deterministic, replayable. Recomposition later would see drifted state; the store-at-emit design is load-bearing and documented in the tests
- **Motive analysis** per utterance, from both agents' state: `premeditated` (high resentment + pride, public delivery), `venting`, `genuine`, `performative` (charity with an audience), `deceitful` (mutated retellings), `bonding`, `desperate` — each with a human summary and the numeric reasons (relationship percentages, trait values, witness counts)
- Categories for bubble coloring: chat / rumor / hostile / kind / status / alarm

## Verified
- `tests/sim/speech.test.ts` (3): lines + motives attach across a 12-day run, identical across same-seed runs, motive kinds match relationship state (insults cite resentment).

## Deferred to stages 23/24 proper (and 29/30)
- Thought bubbles are renderer-derived (needs + last trace) — already planned as presentation-layer
- Items/relics/newspapers/books as phrase sources once those exist (stages 25, 41); qwen3:4b dialogue flavor replaces templates for high-value moments at stage 30
