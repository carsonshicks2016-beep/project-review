# M3 — Self-play duels

**Goal:** two learning agents fight each other; skill emerges from self-play; ML-Agents tracks
**ELO**. This is the payoff milestone.

## New script
- `MatchManager.cs` — owns two `DuelAgent`s, resets them, detects win (death / ring-out /
  timeout), assigns terminal rewards, ends both episodes together, wires opponent references.

## Wiring
1. Build the arena as a **ring** (a platform with edges so fighters can be knocked off).
2. Place **two fighters** (each = ragdoll + `DuelAgent` + `Health` + `Weapon` + a
   `MotionLibrary`/reference clone). They are identical.
3. **Behavior Parameters on BOTH must share the same Behavior Name** (`Duelist`) — that's how
   ML-Agents self-play pits them against each other. Set **Team Id = 0** on one, **Team Id = 1**
   on the other.
4. Add a `MatchManager`; assign `fighterA`, `fighterB`, `spawnA`, `spawnB` (opposite sides),
   and set `ringRadius` to the platform. It auto-wires opponents in `Start()`.
5. Remove the `TrainingDummy` — the opponent is now a real agent.

## Train with self-play
```bash
mlagents-learn config/duel_ppo_selfplay.yaml --run-id=duel01 --time-scale=20 --no-graphics
```
The `self_play` block (already in that config) snapshots past opponents and tracks ELO.
Watch **Self-play/ELO** climb in TensorBoard — that's your skill curve.

## Throughput
Duplicate the whole ring (both fighters + MatchManager + reference clones) 8–16×.

## What to look for (emergent behavior)
Early: flailing, both fall. Mid: they face off and swing. Later: spacing, blocking on read,
punishing whiffed swings, ring-out pushes. Save checkpoints — the gen-1-vs-gen-N replay is the
best thing you'll show off (see `ReplayRecorder` + `docs/M5_RESEARCH.md`).

## Tuning
- **Both just turtle (block forever)** → raise `attackStaminaCost` of blocking less, add a small
  per-step time penalty, or reward aggression slightly.
- **Suicidal trading** → lower `wDealt` relative to `wTaken`.
- **Matches never end** → check ring-out bounds and `maxMatchSeconds`.
- **ELO flat** → self-play `swap_steps`/`save_steps` too large for your step rate; lower them.

## When duels look good
You have a working fighter. Next: `docs/M4_AMP.md` for natural motion (so they stop looking
stiff), and `docs/M5_RESEARCH.md` for tournaments, ablations, and replays.
