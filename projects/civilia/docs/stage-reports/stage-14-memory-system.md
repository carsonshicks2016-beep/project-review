# Stage 14 — Memory System

**Status: complete** — 2026-06-11

## Delivered
- `Memory` model + `memorySystem.ts`: each agent's **personal, fallible, first-person record** of what mattered to them — the event log stays canonical; memories are what agents act on
- **Selective formation**: insults brand the target (harder when public, harder still for the proud), thefts mark the thief (shame if caught, fear if not) and the witnesses (paranoia makes them unforgetting), kindness warms, apologies land as forgiveness or a second humiliation, rumors arrive as hearsay, life events (founding a household, finishing a building) burn bright — and small talk, meals, and work shifts are lived but never remembered
- **Tone + importance + emotional charge** per memory (warm/proud/angry/ashamed/afraid/worried/bitter); first-person template summaries (qwen3:4b will reword, not restructure, at stage 30)
- **Nightly decay**: strength fades scaled by importance and emotional heat; below the floor the memory is gone for good. 36-memory cap per head — a full head keeps what burns brightest
- **Peek vs. recall**: decision scoring uses read-only `peekMemories`; acting on a memory (`recallMemories`) strengthens it. Lashing out dwells on the wound — which produced an emergent grudge spiral in the 30-day probe (one humiliation recalled 31×, strength 0.98)
- **Memories bias decisions** with visible reasons: remembered kindness boosts helping ("remembers: 'Mara gave me bread when I was going hungry'") and talking; remembered wounds boost striking back ("still remembers: '…insulted me in front of everyone'")
- **Agent memory panel**: "What they remember" in the drawer — tone emoji, summary, day, kind, dwell count, strength bar
- `SAVE_VERSION` 3 (persistence test now tracks the constant)

## Verified
- `tests/sim/memory.test.ts` (5): formation asymmetry (target ≫ insulter ≫ small-talk none), trivial-fades-before-searing decay, peek/recall semantics, memory-biased insult scoring with reason strings, bounded + deterministic accumulation with every memory tracing to a logged event. Suite 66/66, tsc clean, panel verified in-browser.

## Known characteristics (deliberate, revisit at stage 18)
- All agents hit the 36-memory cap by ~day 30: the bread-crisis town generates heavy social traffic, so cap-pruning does more forgetting than decay does. Tuning town mood is economy work (stage 18), not memory work.
- 'Ashamed' dominates the tone census (insult-heavy dynamics) — same root cause.
