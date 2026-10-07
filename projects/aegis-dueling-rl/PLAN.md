# Aegis — Physics-Based Sword & Shield Combat RL

> A 3D active-ragdoll humanoid wielding a sword and shield learns to **duel** through
> reinforcement learning and self-play. Built for **research/learning**, not just a demo.

**This document is authoritative.** Update it as decisions change.

- **Started:** 2026-06-15
- **Engine / stack:** Unity + ML-Agents (C# environment, Python PPO/self-play trainer)
- **Control model:** Physics ragdoll — torque/PD joint control (ASE / DeepMimic style)
- **Primary goal:** RL research & learning (clean experiments, ablations, ELO)

---

## 1. North star & references

This is essentially a hobby-scale reimplementation of:

- **ASE** — *Adversarial Skill Embeddings* (Peng et al., SIGGRAPH 2022). The demo character is
  literally a sword-and-shield warrior. This is the target aesthetic/capability.
- **DeepMimic** (Peng et al. 2018) — physics characters imitate mocap clips via RL. The
  technique that gets a ragdoll to move *naturally* instead of flailing.
- **AMP** — *Adversarial Motion Priors* (Peng et al. 2021) — natural motion from a dataset via
  a discriminator reward, without locking to one clip. The M4 upgrade.
- **Control Strategies for Two-Player Competitive Sports** (Won et al. 2021) — physically
  simulated boxing & fencing via self-play. The closest prior art to the *fighting* part.

---

## 2. The core insight (and biggest risk)

**Physics motor control gates "fighting" behind "balancing."** A torque-controlled humanoid
with no guidance just collapses. Learning to win a swordfight from scratch is hopeless because
the agent never gets far enough to discover combat.

**The unlock: motion imitation.** Reward the agent for matching reference animation clips
(idle → walk → strike → block), so it learns natural, balanced movement first. Then blend in
the *task* reward (hit the opponent, don't get hit). This is the DeepMimic → AMP progression.

**Consequence for the roadmap:** the critical, make-or-break milestone is **M1 (balance &
locomotion via imitation)**, not the fighting. Most hobby attempts die there. We de-risk it
early and lean hard on reference data rather than learning from zero.

---

## 3. Architecture

```
┌─────────────────────────── Unity (C#) ───────────────────────────┐
│  Arena prefab  ×N  (parallel for throughput)                       │
│   ├─ Agent A (TeamId 0)        ┐                                    │
│   │   ├─ ArticulationBody rig  │  active ragdoll, PD-driven joints  │
│   │   ├─ Sword + Shield        │  hitboxes / hurtboxes              │
│   │   ├─ BehaviorParameters    │  obs/action spec                   │
│   │   └─ DecisionRequester     │  act every K physics steps         │
│   └─ Agent B (TeamId 1)        ┘                                    │
│  Reference Animator (kinematic) → sampled target pose per frame     │
└────────────────────────────────────────────────────────────────────┘
              ▲ observations / actions / rewards │
              │                                  ▼
┌──────────────────────── Python (ML-Agents) ───────────────────────┐
│  mlagents-learn  →  PPO  +  self_play (snapshots, ELO)              │
│  TensorBoard logs · YAML configs · curriculum · checkpoints         │
└────────────────────────────────────────────────────────────────────┘
```

**Key technical choices**

- **`ArticulationBody`, not `ConfigurableJoint`.** Reduced-coordinate articulation is far more
  stable for chained joint motor control. Use `ArticulationDrive` (stiffness/damping/target)
  per DoF for PD position control. Pelvis = floating articulation root.
- **Action = target joint angles** (PD targets), not raw torque — much easier to learn and more
  stable. Consider Stable-PD (DeepMimic) later if jitter is a problem.
- **Self-play via teams.** Agent on TeamId 0 vs TeamId 1; ML-Agents handles opponent snapshots
  and **ELO** automatically when `self_play` is enabled.

---

## 4. Combat model

Borrowed from fighting games + Mordhau/Chivalry:

- **Attacks** have windup → active → recovery phases (so they're punishable).
- **Attack types:** light / heavy / thrust / overhead — speed vs damage/reach tradeoffs.
- **Shield:** directional block with a coverage arc; block costs stamina; shield bash.
- **Stamina & poise:** attacking/blocking drains stamina; hits cause stagger.
- **Hitboxes/hurtboxes** on the physics bodies; sword *tip* velocity gates damage.
- **Win condition:** HP to zero, or ring-out (knocked/pushed off the arena).

Damage is physics-aware (impact velocity + location), which is the point of going ragdoll.

---

## 5. RL design

**Observations** (root/local-relative for invariance):
- Own joint angles + angular velocities, root up-vector & velocity (balance), center of mass.
- End-effector + sword-tip positions/velocities relative to root.
- Opponent relative pose, velocity, and current action phase (to react/punish).
- Stamina, HP, contact flags. Optionally a short history buffer.

**Actions:** continuous — normalized PD target per actuated DoF.

**Reward = imitation + task + regularizers**
- *Imitation* (DeepMimic): exp-weighted pose / velocity / end-effector / COM match to the
  reference clip for the current phase. Dominant early.
- *Task:* damage dealt (+), damage taken (−), win bonus, stay-upright bonus.
- *Regularizers:* torque/energy penalty (kills jitter), symmetry/effort.
- Shift weight from imitation→task as training progresses (or use AMP at M4).

**Self-play & curriculum**
- ELO-tracked self-play with an opponent snapshot pool (built into ML-Agents).
- Curriculum (ML-Agents Environment Parameters): stationary dummy → scripted bot →
  full self-play; enable the opponent only after balance is solid.

---

## 6. Milestones

| # | Milestone | Done when… |
|---|-----------|-----------|
| **M0** | **Viewer + active ragdoll** | Unity scene, humanoid (Mixamo) imported, `ArticulationBody` rig with sword+shield, camera, arena. Can apply PD targets and it moves. ML-Agents toolchain trains a trivial dummy reward end-to-end. |
| **M1** | **Balance & locomotion via imitation** ⚠️ critical | Ragdoll learns to **stand, balance, walk & turn** under torque control by imitating reference clips. *If this fails, nothing downstream works.* |
| **M2** | **Combat actions + dummy** | Adds slash/block/hit-react imitation; hitboxes, HP, stamina. Agent strikes a training dummy and raises shield on cue while staying upright. |
| **M3** | **Self-play duels** 🏆 | Two ragdoll agents, ML-Agents self-play + ELO. Task reward = damage dealt − taken + win. **Emergent dueling** (spacing, blocking, punishing). |
| **M4** | **Style & robustness (AMP)** | Replace frame-locked imitation with an **adversarial motion prior** so fighters improvise natural moves. Add weapon/body variation, domain randomization, league training. |
| **M5** | **Research polish & stretch** | Ablations, ELO ladder + tournament, replay capture, morphology experiments, human-vs-AI, optional LLM commentator. |

---

## 7. Throughput on a Mac (Apple Silicon)

No NVIDIA GPU → **Isaac Gym is out** (CUDA-only). Mitigations:
- **Parallel arenas:** duplicate the arena prefab 8–32× in one scene.
- **Crank time:** `mlagents-learn --time-scale 20`, `--no-graphics`, headless build,
  `--num-envs N` for multiple instances.
- Physics is CPU-bound here; the policy MLP is small and trains fine on CPU/MPS.
- Tune `Time.fixedDeltaTime` + solver iterations — sword impacts create large impulses.

---

## 8. Research practices (since the goal is learning)

- **TensorBoard** for everything (ML-Agents logs natively): reward components, episode length,
  ELO over time, value loss.
- **Configs as code** — every experiment is a versioned YAML in `config/`.
- **Reproducibility** — fixed seeds; record git SHA + config per run.
- **Ablations** — imitation on/off, AMP on/off, per-reward-term contribution, observation
  dropout (e.g., hide shield state). Keep a lab notebook of hypotheses & results.

---

## 9. Expansion ideas (backlog)

- Weapon roster (dagger / greatsword / spear / bow) → asymmetric matchups.
- Co-evolve **body morphology + controller** (Karl Sims style).
- **Quality-Diversity (MAP-Elites)** for a roster of distinct styles, not one optimal blob.
- Replay system + gen-1-vs-gen-N side-by-side (the satisfying "before/after").
- Tournament bracket, ELO ladder, named fighters with auto "personalities".
- Human-vs-AI; export policy (ONNX) so fights run anywhere.
- LLM commentator / coach (defer — late stretch).

---

## 10. Open questions

- [ ] Source rig: Mixamo Y-bot vs a custom humanoid model? (Mixamo gives free combat anims.)
- [ ] Reference dataset: which clips, and how to phase-align them to the imitation reward?
- [ ] AMP at M4 — extend/fork ML-Agents, or implement the discriminator as a custom reward
      signal? (ML-Agents has no built-in AMP.)
- [ ] Project name — "Aegis" (= shield) is a placeholder; rename freely.

---

## 11. References

- ASE: https://xbpeng.github.io/projects/ASE/
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
- AMP: https://xbpeng.github.io/projects/AMP/
- Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
- ML-Agents self-play docs: search "ML-Agents Training-Self-Play"
- Mixamo (free rigged characters + combat anims): https://www.mixamo.com
