# FOLDSPACE

A flat strip learns to **fold itself into shapes**. A PPO policy creases a chain
of hinged segments until its silhouette matches a target — a circle, a star, a
heart — scored by how closely it traces the outline. A web viewer animates each
fold over its ghosted target.

---

## Quick start

```bash
# 1. train — ~12 min on a laptop CPU
python3 -m foldspace.train --iters 700 --envs 128 --rollout 96 --out foldspace/checkpoints/fold.pt

# 2. record one fold per shape into the viewer
python3 -m foldspace.record --ckpt foldspace/checkpoints/fold.pt

# 3. watch (served by "fold-viewer" in .claude/launch.json, or:)
cd foldview && python3 -m http.server 8798   # open http://localhost:8798
```

Controls: **space** play/pause · **← / →** prev/next shape · scrubber · speed ·
**AUTO** cycles the gallery · **GHOST** toggles the target outline.

---

## How it works

### The strip (`env.py`)
The strip is a chain of **K=28 equal segments** with a fold angle at each joint.
The policy outputs a *target* fold configuration (a base heading + 27 joint
angles); the strip **eases toward it at a bounded rate**, so it physically folds
over the episode while each episode's return cleanly reflects one configuration's
quality.

### The reward — symmetric Chamfer distance
Both the folded strip and the target outline are sampled to M=96 points. Reward
each step is the **reduction in symmetric Chamfer distance** between them:

```
chamfer = mean_i min_j |strip_i - target_j|   +   mean_j min_i |strip_i - target_j|
```

The two directions matter: the first pulls every strip point onto the outline;
the second forces the strip to *cover the whole outline* (so it can't cheat by
bunching up). It's translation-invariant (both clouds are centred), needs no
rasterisation, and telescopes — episode return = `initial_chamfer − final_chamfer`,
so maximising it maximises final shape fidelity. A `score = 1 − chamfer/chamfer₀`
in [0,1] is reported for readability.

### The shapes (`shapes.py`)
A gallery of parametric silhouettes — circle, square, triangle, star, heart,
spiral, cross, crescent — each normalised so its perimeter equals the strip
length (so a perfect trace is achievable). The policy is told which target via a
one-hot id. An **oracle** (each shape's own discrete curvature) drives a built-in
sanity check: setting the strip to the oracle drops Chamfer ~30× vs a straight
strip, proving the strip can represent every shape.

### The lesson that made it learn
The first formulation had the policy emit *incremental* per-joint nudges. It
plateaued at ~60%: independent per-joint noise can't discover the **coordinated**
curls that simple shapes need (a circle needs all 28 joints to agree), so wiggly
shapes learned and a circle stuck at 31%. Switching to **target-configuration
output + bounded easing** gives each episode a clean whole-config reward signal —
circle and square jumped to ~88%, and every shape lands in the 70–90% range.

### The policy (`networks.py`, `train.py`)
A Gaussian actor-critic MLP, trained with PPO (reusing `duel/ppo.py` — GAE,
clipped surrogate, annealed entropy). Single-agent, one stream per arena.

### The viewer (`foldview/`)
A blueprint/origami canvas renderer: graph-paper grid, the target ghosted as a
dashed glowing outline, and the strip drawn as a gradient ribbon (cyan head →
magenta tail) with glowing crease nodes that fold into place over the target.

```
foldspace/
  shapes.py     parametric target silhouettes + oracle
  env.py        vectorised folding chain + Chamfer reward
  networks.py   Gaussian actor-critic
  train.py      PPO training (reuses duel.ppo)
  record.py     fold each shape -> JSON clips
foldview/       index.html / style.css / app.js  + replays/
```
