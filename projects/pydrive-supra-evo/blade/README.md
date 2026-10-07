# BLADE — curriculum 3D melee

Articulated MuJoCo humanoids learn to **stand, walk, and fight** with sword,
shield, mace, axe, katana — or bare fists — on uneven terrain, trained by
self-play PPO through an automatic **curriculum** and watched in a clean live 3D
viewer.

This is a ground-up rebuild around the lesson that end-to-end "stand + walk +
fight all at once" reward-hacks into degenerate poses. Instead the agent
graduates one skill at a time, each gated by a competence test.

---

## Quick start

```bash
pip install mujoco imageio imageio-ffmpeg pygame      # one-time

# 1. train a fighter through the curriculum (lab-grade headless dashboard)
python3 -m blade.train --name knight --loadout sword_shield

# 2. watch it live in clean 3D (hot-reloads, follows the curriculum)
python3 -m blade.view  --name knight

# 3. pit any two trained fighters — even different weapons
python3 -m blade.arena --left knight --right reaper          # -> mp4
python3 -m blade.arena --left knight --right reaper --live   # 3D window
```

**Loadouts:** `none` · `sword` · `shield` · `sword_shield` · `mace` · `axe` · `katana`
(each with realistic mass/inertia — the mace is head-heavy at ~3.4 kg, the katana
light at ~1.1 kg).

---

## The curriculum

| Stage | World | Learns | Graduates when… |
|---|---|---|---|
| **STAND** | flat, far apart | balance & stand tall | upright ≥ 86% of the episode |
| **WALK** | flat | locomote toward the foe | closes ≥ 70% of the gap |
| **TERRAIN** | uneven | keep it up on rough ground | closes ≥ 60% of the gap |
| **SPAR** | uneven, close | armed combat | (final stage) |

Each rung is a `StageConfig` ([config.py](config.py)) — world, reward weights,
episode rules, and a promotion threshold. A greedy **evaluation**
([curriculum.py](curriculum.py)) measures competence (survive / reach / hits)
each cycle; when the bar is cleared the trainer **auto-promotes**, rebuilds the
world for the next rung, and carries the policy over. The whole point: standing
isn't "torso above 0.55 m" (which a sprawl games) — it's a real, tested skill
before combat is ever introduced.

## Why it's robust

- **Residual control:** action 0 holds a guard stance; the policy learns
  deviations — a balanced starting point, not flailing from a heap.
- **Weapon-agnostic core:** any armament's striking surface is tagged `strike_*`
  with a `weapon_tip` site, so observations, the aim reward, and hit/block
  detection work identically for fist, sword, mace, axe, katana, or shield.
- **Self-play PPO** (reuses `duel/ppo.py` + `foldspace/networks.py`), MuJoCo
  threaded across cores via `VecBlade`.

## Lab-grade headless output

```
┏━━━ B L A D E   curriculum trainer ━━━ lab ┓
  run knight   loadout sword_shield   device cpu   budget 20M steps
  curriculum  STAND → WALK → TERRAIN → SPAR

   iter │ stage        │ promotion      │ throughput        │ instruments
    240 │ S1/4 STAND   │ ████████░░ 82% │ 12.3M 7.9k/s ETA 9m │ surv 0.79 reach .. ret/s +0.94 ent 6.1 kl 0.018
  ════ ✓ PROMOTED  STAND → WALK  ·  iter 255  ·  survive 0.87 ════
```

## The simplified viewer

`blade.view` renders MuJoCo **offscreen and blits to a clean pygame window** — no
MuJoCo debug panels, no `mjpython` — with a minimal HUD (loadout, stage gauge,
health/survival), hot-reloading the checkpoint and switching worlds as the agent
is promoted.

```
blade/
  weapons.py    7 loadouts (fist/sword/shield/mace/axe/katana) w/ real mass
  model.py      MJCF humanoid, per-fighter loadout, flat/heightfield terrain
  config.py     StageConfig + the default curriculum ladder
  env.py        stage-driven BladeEnv (+ threaded VecBlade)
  curriculum.py stage manager + greedy competence evaluation
  dashboard.py  lab-grade console
  train.py      curriculum self-play PPO trainer
  view.py       simplified live 3D viewer (offscreen->pygame)
  arena.py      matchmaker: any two fighters, any loadouts -> mp4 / live
  render.py     tracking camera + mp4 writer
```
