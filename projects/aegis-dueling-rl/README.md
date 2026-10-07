# Aegis

A 3D physics-ragdoll humanoid with a **sword and shield** that learns to duel via
reinforcement learning and self-play. Research-focused.

- **[PLAN.md](PLAN.md) is the source of truth** — read it first.
- Stack: **Unity + ML-Agents** (C# env, Python PPO/self-play).
- Control: **physics ragdoll** (ArticulationBody + PD targets, ASE/DeepMimic style).

## Status
Planning (2026-06-15). No Unity project created yet — see PLAN.md §6 milestones.

## Next steps
Follow **[docs/M0_SETUP.md](docs/M0_SETUP.md)** — the step-by-step M0 checklist
(toolchain, Mixamo rig + sword/shield, arena, first-pass ragdoll).

## Layout
- `docs/M0_SETUP.md` — M0 setup checklist (start here).
- `config/duel_ppo_selfplay.yaml` — starter ML-Agents trainer config (used at M3).
