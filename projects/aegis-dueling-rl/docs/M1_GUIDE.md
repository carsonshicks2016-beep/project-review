# M1 — Wiring & Training Guide

Code is written (`Assets/Scripts/`). This is how you wire it in the Unity editor and train
the ragdoll to **balance & track a reference motion**. Do M0 first.

> Reality check: physics rigs never work first-try. Expect to tune joint axes, drive gains,
> limits, and reward weights. The goal of M1 is a humanoid that stays upright and roughly
> matches an idle/walk clip — not perfection.

## Scripts
- `RagdollBuilder.cs` — builds the `ArticulationBody` rig + colliders from a Humanoid Animator.
- `PDJointController.cs` — maps actions → PD drive targets; exposes joint state + action size.
- `ReferencePoseProvider.cs` — kinematic clone that plays the reference clip to imitate.
- `DuelistAgent.cs` — the ML-Agents `Agent`: observations, actions, imitation+upright reward.

## Step 1 — Build the ragdoll
1. Select your imported Mixamo character (Rig = **Humanoid**) in the scene.
2. Add **RagdollBuilder**, then in its component context menu (⋮) click **Build Ragdoll**.
3. Press Play — it should flop believably. If it explodes/spasms: lower stiffness or raise
   damping in the generated `ArticulationBody` drives; check colliders don't overlap.
4. **Disable the ragdoll's Animator** (uncheck the component, or set Update Mode so it doesn't
   overwrite the physics pose). The script still resolves bones from it.

## Step 2 — Reference clone
1. Duplicate the character; remove its RagdollBuilder + all `ArticulationBody`/`Collider`s.
2. Add **ReferencePoseProvider**; assign its `Animator` and a `clip` (start with **idle**).
3. Hide its renderers (it's logic-only). Place it anywhere — only bone rotations are read.

## Step 3 — Agent object
On the ragdoll root, add: **DuelistAgent**, **PDJointController**, **Decision Requester**,
**Behavior Parameters** (all from ML-Agents).
- `PDJointController.root` = pelvis `ArticulationBody`.
- `DuelistAgent`: assign `controller`, `reference`, `ragdollAnimator` (the ragdoll's Animator),
  `root` (pelvis).
- **Decision Requester**: Decision Period ≈ 4–5, Take Actions Between Decisions = on.

## Step 4 — Match observation/action sizes
1. Press Play once. The Console logs:
   `Behavior Parameters → Continuous Actions = N, Vector Observation Space Size = M`.
2. In **Behavior Parameters**: Behavior Name = `Duelist`, set **Continuous Actions = N** and
   **Vector Observation Space Size = M**. (These must match exactly or training errors.)

## Step 5 — Parallelize
Duplicate the whole arena (ragdoll agent + its reference clone) 8–16× and spread them out.
More arenas = more samples per second.

## Step 6 — Train
From the repo root, in your activated Python 3.10 venv:
```bash
mlagents-learn config/balance_ppo.yaml --run-id=balance01 --time-scale=20 --no-graphics
```
Press **Play** when prompted. Monitor:
```bash
tensorboard --logdir results
```
Success = mean reward and episode length both climbing; visually, it stops face-planting and
starts holding the reference pose.

## Tuning che-sheet
- **Falls instantly** → legs too weak: raise hip/knee `stiffness`; raise `minRootHeight` only
  if it's set above the real standing height.
- **Jitters/vibrates** → stiffness too high or damping too low; raise damping.
- **Bends the wrong way (knee/elbow)** → fix that joint's hinge: set `ArticulationBody.anchorRotation`
  so the X axis aligns with the bend axis, or flip the revolute `lowerLimit/upperLimit` signs.
- **Won't track the clip** → raise `wPose`; confirm the reference actually animates (see below).
- **Reference looks wrong** → `AnimationClip.SampleAnimation` can misbehave on Humanoid clips.
  Alternative: give the clone an AnimatorController with one state containing the clip, and in
  `ReferencePoseProvider.Sample` use:
  `referenceAnimator.Play("Idle", 0, Phase01); referenceAnimator.Update(0f);`

## Optional upgrade — Reference State Init (RSI)
Resetting to a *random phase* of the clip each episode (instead of the spawn pose) speeds
DeepMimic learning a lot. It needs reduced-coordinate joint targets per phase, which is fiddly
with `ArticulationBody`. Get the spawn-pose version learning first; ping me and I'll add RSI.

## When M1 works
Tell me it's balancing/tracking and I'll write **M2**: combat clips (slash/block/hit-react),
hitboxes/hurtboxes, HP/stamina, and a training dummy.
