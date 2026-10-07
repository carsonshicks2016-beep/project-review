# M2 — Combat actions vs a dummy

**Goal:** the agent learns to throw a sword swing that *lands* on a training dummy and to
raise its shield, all while staying upright. Builds directly on the M1 ragdoll.

## New scripts
- `Health.cs` — HP, stamina, poise, block resolution (shield arc). One per fighter.
- `Weapon.cs` — blade hit detection; damage only in the attack ACTIVE window and above a tip
  speed. Put a trigger collider along the sword blade.
- `MotionLibrary.cs` — swaps the reference clip (idle/walk/slash/block/hit) by combat state.
- `DuelAgent.cs` — the full combat agent (replaces `DuelistAgent`): hybrid action space
  (continuous PD targets + discrete intent {none, attack, block}).
- `TrainingDummy.cs` — scripted opponent (Passive / Blocker / Aggressor).

## Wiring
1. On the fighter root: add `Health`. Assign `shieldForward` = the shield transform.
2. On the sword: add a **trigger collider** down the blade + `Weapon`; set `owner` = this
   fighter's Health, `tip` = a transform at the sword point.
3. Replace `DuelistAgent` with `DuelAgent`. Assign `controller`, `motion`, `ragdollAnimator`,
   `root`, `health`, `sword`.
4. On the reference clone: add `MotionLibrary`, assign its `provider` (the
   `ReferencePoseProvider`) and the five Mixamo clips.
5. Build a dummy: a second character with `Health` + `TrainingDummy` (start mode = Passive).
   Assign `DuelAgent.opponent` = dummy transform, `opponentHealth` = dummy Health.
6. Press Play, read the Console for the new **Continuous / Discrete / Observation** sizes, set
   them in Behavior Parameters (now also add a **Discrete Branch** of size **3**).

## Curriculum (recommended)
Train against the dummy in this order — switch the `TrainingDummy.mode` as it improves:
1. **Passive** — learn to land a swing at all.
2. **Blocker** — learn to read the shield / hit when it's down.
3. **Aggressor** — learn to block and trade. (Give the dummy a `Weapon` for this.)

## Train
```bash
mlagents-learn config/combat_ppo.yaml --run-id=combat01 --time-scale=20 --no-graphics
```
Success = damage-dealt climbing while the agent stays upright. Watch in TensorBoard.

## Tuning
- **Never lands hits** → lower `Weapon.minTipSpeed`, lengthen `attackTiming.y` (active window),
  check the blade trigger collider actually covers the edge.
- **Falls over when swinging** → swings destabilize balance; raise leg stiffness or lower
  `wDealt` early so it doesn't suicide-swing.
- **Spams attack** → raise `attackStaminaCost`; ensure stamina regen isn't too fast.
- **Won't block** → confirm `Health.shieldForward` points out of the shield face.

## When M2 works
Tell me it reliably lands hits and blocks → I'll have already given you M3 (self-play). Move to
`docs/M3_GUIDE.md`.
