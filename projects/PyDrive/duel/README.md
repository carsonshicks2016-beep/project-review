# VECTOR DUEL

Two automata duel in a neon void. Nobody scripted how they fight — they learned
it from scratch by playing millions of duels against themselves with **PPO**.

A single policy controls both fighters (symmetric self-play). It learns to
approach, aim, trade fire, dash through bullets, and close out kills — emergent
behaviour that reads as genuinely tactical. A web renderer plays back recorded
matches as a stylised highlight reel.

---

## Quick start

```bash
# 1. train a policy (self-play PPO) — ~5 min on a laptop CPU
python3 -m duel.train --iters 500 --envs 256 --rollout 128 --out duel/checkpoints/duel.pt

# 2. record a batch of duels into the viewer as JSON replays
python3 -m duel.record --ckpt duel/checkpoints/duel.pt --matches 16

# 3. watch the highlight reel (web)
#    served by the "duel-viewer" entry in .claude/launch.json, or just:
cd viewer && python3 -m http.server 8799   # then open http://localhost:8799
```

Web controls: **space** play/pause · **← / →** prev/next duel · scrubber · speed
(0.5/1/2×) · **AUTO** cycles the playlist (ordered as a highlight reel:
decisive fights first, most action first).

### Watch it learn — live (`watch.py`)
A pygame window that renders live duels from the current policy and **hot-reloads
the checkpoint whenever `train.py` writes it**, so the fights visibly sharpen as
training runs. It's fully decoupled — zero impact on training throughput.

```bash
# terminal 1 — train, saving often so the live view updates every few seconds
python3 -m duel.train --save-every 5 --out duel/checkpoints/duel.pt

# terminal 2 — watch
python3 -m duel.watch                 # one large featured duel
python3 -m duel.watch --grid 6        # six simultaneous duels (a training dashboard)
```

Keys: **SPACE** pause · **D** toggle deterministic/stochastic · **R** force-reload
· **ESC** quit. (`--shot out.png` renders a single frame headlessly and exits.)

---

## How it works

### The arena (`env.py`)
A fully **vectorised numpy** environment steps thousands of independent duels in
lockstep (~250k env-steps/s on CPU), which is what makes self-play tractable
without a GPU. Each fighter:

- **thrusts** with continuous 2D acceleration (drag + speed cap),
- **aims** toward a target direction at a limited turn rate (so aiming is a skill
  and feints matter),
- **fires** travelling projectiles on a cooldown (bullets are dodgeable — faster
  than a runner but beatable with a dash),
- **dashes** — a burst of speed with brief i-frames (a real dodge, not a panic
  button).

The policy sees a **39-dim egocentric observation**: its own kinematics and
cooldowns, the opponent's relative state, an explicit *"is the enemy aiming at
me"* signal, wall distances, and the **3 nearest incoming bullets** (position +
velocity) — the sensory basis for dodging.

### The policy (`networks.py`)
A small actor-critic MLP (shared tanh trunk, orthogonal init) with a **mixed
action head**: a Gaussian over the 4 continuous controls (thrust x/y, aim x/y)
and a Bernoulli pair over the 2 binary controls (fire, dash), plus a value head.

### The learner (`ppo.py`, `train.py`, `selfplay.py`)
PPO from scratch — GAE(λ), clipped surrogate, clipped value loss, entropy bonus,
KL early-stop. Each duel yields up to two independent training streams (one per
fighter). Most arenas are pure mirror self-play; a configurable fraction pit the
current policy against a **frozen snapshot from an opponent pool**, which keeps
newly-learned strategies from forgetting how to beat older ones.

### Reward design — the one thing that matters
Naïve symmetric rewards (`+x` for a hit, `-x` for being hit) collapse into a
**mutual-avoidance stalemate**: the safe play is to never engage, so both
fighters kite to a timeout. The fix that produces real fights:

| | value | why |
|---|---|---|
| land a hit | **+0.70** | landing is worth more than being hit costs… |
| take a hit | −0.30 | …so *trading blows* is positive-EV → aggression |
| win / lose | ±1.0 | the duel is still about winning, not just trading |
| timeout draw | **−0.70** | stalling is nearly as bad as losing → no kiting |

Combined with a tighter arena and a 15 s cap, this took the policy from **~15%
decisive, 18 s kite-fests** to **98% decisive, ~3 s aggressive duels** (balanced
51/47 between the two sides over 512 eval matches).

### The renderer (`viewer/`)
A single-file canvas engine: high-DPI, frame-interpolated playback, additive
bloom, motion trails, comet-tail tracers (matched across frames), muzzle flashes,
hit sparks, expanding shockwaves, screen-shake, and a slow-mo verdict card. The
two fighters — **ARC** (cyan) and **HEX** (rose) — are drawn as angular
arrowheads with health rings, aim-laser sights, dash streaks, and i-frame
shields. The replay format embeds all arena constants, so the viewer always
matches the sim it came from.

---

## Files
```
duel/
  env.py        vectorised 2-player arena + projectiles + observations
  networks.py   actor-critic with a mixed Gaussian/Bernoulli action head
  ppo.py        GAE + clipped PPO update
  selfplay.py   frozen opponent pool
  train.py      self-play training loop
  record.py     play matches -> JSON replays + highlight manifest
  watch.py      live pygame viewer with checkpoint hot-reload (view while training)
viewer/
  index.html / style.css / app.js   the VECTOR DUEL renderer
  replays/      recorded matches (regenerated by record.py)
```
