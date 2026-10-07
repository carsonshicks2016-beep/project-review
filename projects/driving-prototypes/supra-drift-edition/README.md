# 🏎️ Supra-AI — DRIFT EDITION

> **This is the dedicated drift fork.** It's a complete, independent copy of the
> project for developing the **drift specialist** (`--drift` / `--watch-drift`).
> The traditional fast-lap racer lives in the separate `supra-ai/` project. The
> two share no files and no checkpoints — work here can't affect the racer and
> vice-versa. The race commands (`--ppo`) still exist here but are not the focus.
>
> Drift uses its own everything: reward, observation (30-d, with sustain/chain/
> gutter/grade state), checkpoints (`drift_supra.pt`), track pool, episode
> budget, exploration, steering lock, and a handbrake. See the drift section
> below.

A neuro-evolution driving simulator. A population of **Mk4 (A80) Toyota Supras**
— each with its own little neural-network brain — learns to drive a procedurally
generated twisty circuit. The fittest brains breed and mutate every generation,
so the field visibly gets faster, smoother, and starts to *drift* on its own.

The physics is a real **four-wheel** vehicle-dynamics model (not arcade):
four independent contact patches, lateral **and** longitudinal weight transfer
with suspension lag, load-sensitive nonlinear tyres, slip-ratio longitudinal
grip with real wheel inertia (so it spins up and locks), a limited-slip diff,
a 2JZ-GTE torque curve, sequential twin-turbo lag, and a 6-speed gearbox. The
driver inputs are rate-limited like a human's, with self-aligning torque fed
back as steering feel. There's live engine + turbo sound pitched to RPM, and a
side dashboard that shows *everything* — including a live 4-corner load diagram.

---

## Run it

```bash
cd supra-ai
python3 run.py                 # watch a population evolve live (with sound)
python3 run.py --train 300     # fast, headless: evolve 300 generations, save best
python3 run.py --watch         # load the saved champion (GA) and watch it drive
python3 run.py --ppo 1500      # train a PPO policy (parallel, headless) -> fast lap
python3 run.py --ppo 1500 --workers 8   # cap the rollout worker processes
python3 run.py --watch-ppo     # watch the trained PPO policy drive
python3 run.py --drift 1500    # train a DRIFT-objective policy (separate lineage)
python3 run.py --watch-drift   # watch the trained drift policy slide
python3 run.py --ppo 2000 --recurrent --ppo-save ppo_lstm.pt   # LSTM policy (memory)
python3 run.py --car rx7 --ppo 2000 --ppo-save rx7.pt          # train a different chassis
python3 run.py --car rx7 --watch-ppo --ppo-save rx7.pt         # watch it (pass the SAME --car)
python3 run.py --seed 42       # fixed track + RNG (reproducible)
python3 run.py --pop 40        # bigger field
```

> **Note:** the PPO observation now includes a look-ahead **track preview**, so
> any policy/GA brain saved before this change is incompatible. The trainer
> detects this, backs the old file up, and starts fresh — just retrain
> (`--ppo` / `--train`). The path to a genuinely fast lap is a long `--ppo` run:
> it keeps getting quicker the more iterations you give it.
>
> **Checkpoints:** `--ppo` writes two files — `ppo_supra.pt` is the *latest*
> policy (overwritten every 25 iters), and `ppo_supra.best.pt` is the
> *best-so-far*, kept automatically (stage-aware: a higher curriculum stage wins,
> then highest rolling-average distance within a stage). Watch or restore the
> best with `--watch-ppo --ppo-save ppo_supra.best.pt`.

(On macOS the interpreter is `python3`, not `python`.)

Requires: `python3`, `pygame`, `numpy`, `torch` (PPO trainer), `sounddevice`
(optional — sound auto-disables if absent).

### Controls (live mode)

| Key | Action |
|-----|--------|
| `SPACE` | pause |
| `F` | focus next car |
| `A` | toggle auto-follow-the-leader |
| `C` | cycle camera (chase → cinematic → drift cam → broadcast) |
| `V` | toggle auto slow-mo (bullet-time on big slides) |
| `M` | mute |
| `T` | fast-forward (8×) |
| `R` | new random track |
| `+` / `-` | zoom |
| `S` | save the best brain |
| `ESC` | quit |

**Cinematic cameras (`C`):** *chase* (top-down, follows the car) → *cinematic*
(rotates so the car points up-screen, leads into corners) → *drift cam* (aligns
to the car's travel direction so a slide visibly crabs sideways) → *broadcast*
(fixed trackside vantages that cut as the car sweeps past). `V` arms auto
slow-mo, which briefly drops to 0.35× time on a big slide. (These apply to the
watch/evolve views; the live-PPO window keeps its own camera.)

---

## How it works

| File | Role |
|------|------|
| `supra/config.py` | every tunable number — Supra spec, tyres, turbo, gearbox, GA, render |
| `supra/physics.py` | dynamic bicycle model: Pacejka tyres, friction circle, weight transfer, 2JZ + turbo + gearbox |
| `supra/track.py` | procedural circuit, raycast "vision", arc-length progress, difficulty metering |
| `supra/brain.py` | numpy MLP genome (fast, mutable), 6 control outputs |
| `supra/evolution.py` | genetic algorithm: elitism, tournament selection, crossover, mutation |
| `supra/ppo.py` | PPO trainer (actor-critic, obs-normalised, LR-annealed, KL-guarded) + `TorchBrain` |
| `supra/ppo_env.py` | torch-free env (time-budget episodes, progress reward) + parallel rollout workers |
| `supra/simulation.py` | couples car + brain + fitness; shared `observe()`; runs the field around the track |
| `supra/sound.py` | real-time engine harmonics + turbo whine + blow-off, synthesised from RPM |
| `supra/dashboard.py` | the everything-panel (telemetry, brain, fitness, road, minimap) |
| `supra/app.py` | pygame loop, chase camera, smoke + skid marks, the evolution loop |

**What the brain sees (inputs):** 9 raycast distances to the track edges +
longitudinal/lateral velocity, yaw rate, body slip angle, RPM, gear, lateral-G,
and current steering + a **look-ahead track preview** (signed centreline
curvature sampled at 6 distances ahead). The preview is what lets a
feed-forward policy *plan* braking points and apexes instead of only reacting.
**What it controls (outputs):** steering, throttle, brake, **clutch**, and
**sequential up/down shifts** — the brain manages the whole drivetrain itself,
so it learns to keep the 2JZ in the boost and can clutch-kick into drifts.

**GA fitness:** distance travelled along the centreline (across laps) plus a
small reward for carrying speed; cars die off-track or when stalled.

**Pick your chassis (`--car`).** Three presets — `supra` (Mk4 / 2JZ-GTE, the
default), `rx7` (FD3S / 13B-REW: lighter, ~50/50, high-revving and tail-happy),
and `skyline` (R34 / RB26DETT: heavier, front-biased, torquey; modelled RWD
since the sim is rear-drive). Same tyre model; what differs is mass, balance,
power curve, redline and gearing — so a policy trained on each learns a visibly
different style. Pass the same `--car` when training and when watching, and give
each its own save file.

**Two PPO lineages — race & drift.** `--ppo` trains for *lap time*; `--drift`
trains a completely separate policy (`drift_supra.pt`) on a *flipped* reward —
speed × sideways-ness inside a slip window (with a small progress term so it
slides *through* corners, not donuts, and a spin-out penalty so it stays
controlled). Same physics and network; opposite driving style. The two lineages
never share weights, so you can watch a grip brain and a drift brain diverge.

**PPO reward (the fast-lap objective):** episodes run on a fixed *time* budget
and the reward is **centre-line progress per frame** (minus a small smoothness
penalty). Because the budget is fixed, "more progress" literally means "more
speed" — so optimising the return produces a *fast* lap, not merely a complete
one. Hitting the time budget is a **truncation** (the value is bootstrapped);
leaving the track or stalling is a true **terminal** with a penalty, so the
policy learns exactly where the limit is. Observations are running-normalised
and tracks are domain-randomised across a curriculum, so it learns to drive
*any* track rather than memorise one.

---

## 🧠 Ideas to make it cooler / more powerful (roadmap)

### Smarter brains
- ✅ **Done — Reinforcement learning (PPO), tuned for a fast lap.** `supra/ppo.py`
  trains an actor-critic policy across many random tracks (`--ppo`) and plays it
  back live (`--watch-ppo`). The reward is **centre-line progress on a fixed
  time budget** (so the objective is *speed*, not lap-count), with proper
  truncation-vs-termination value bootstrapping, running observation
  normalisation, a look-ahead curvature preview in the observation, LR
  annealing, KL-guarded updates, and **parallel rollout workers**. This is the
  path to *superhuman* — it keeps getting faster the longer you train it.
- ✅ **Done — Recurrent (LSTM) brains.** Opt-in with `--recurrent`: an encoder
  MLP → LSTM → actor/critic heads, trained with proper recurrent PPO (hidden
  state persists across rollouts, resets at episode boundaries, BPTT minibatched
  over environments) and stateful inference (memory carried while you watch,
  cleared on respawn). Memory lets it anticipate corners for smoother
  trail-braking. The feed-forward net is still the default; use a separate save
  file for the LSTM lineage (`--ppo-save ppo_lstm.pt`).
- ✅ **Done — brain controls clutch + gears.** It learns to keep the 2JZ in the
  boost and can clutch-kick into drifts.
- ✅ **Done — track randomisation + curriculum.** Tracks rotate so the AI learns
  to *drive*, not memorise a layout. Tiers: `easy → loop → technical → complex →
  named circuits` (speedway/club/grand-prix). PPO resamples a track *every
  episode* (domain randomisation) and auto-unlocks the next tier once it
  reliably completes laps; GA rotates every couple generations. See
  `supra/track.py` (`STAGES`, `make_track`) and `CurriculumSpec` in config.
- **Real circuits.** Drop non-self-intersecting real layouts (Monaco-style) in
  as waypoint lists via `make_track(control_points=...)`.
- **Imitation bootstrap.** Record a few human laps (arrow keys), pre-train the
  net to clone them, *then* let RL/evolution surpass you.

### Richer physics
- ✅ **Done:** 4-wheel model, lateral + longitudinal weight transfer, suspension
  lag, load-sensitive tyres, slip-ratio longitudinal grip + wheel inertia
  (wheelspin/lockup), limited-slip diff, self-aligning torque, human-rate inputs.
- **Full suspension** (independent spring/damper per corner, anti-roll bars,
  squat/dive geometry) instead of the current lagged transfer.
- **Tyre temperature & wear** — grip that builds then fades over a stint.
- **Surface & weather:** rain lowers `tyre_mu`, puddles, kerbs, dirt run-off,
  changing grip mid-corner. (`mu_scale` is already plumbed into `physics.step`.)
- **Aero downforce** that grows with speed (more grip in fast corners).
- **Fuel load** shifting mass and balance over a race.

### Better racing
- **Head-to-head races** — many brains on track at once, fitness = beating
  rivals, with collision/contact physics. Drafting/slipstream.
- **Overtaking & defending** as learned skills (reward position gained).
- **Ghost of the all-time best** to race against.
- **Championship mode** — points across a season of different procedural tracks.
- **Hall of fame replays** — save + rewatch the best lap ever.

### Spectacle / polish
- **Better sound:** stereo + doppler as cars pass, gear-change blip, tunnel
  reverb, distinct intake vs exhaust, multi-car engine mix.
- **Drift scoring** — angle × speed × duration, smoke intensity tied to slip.
- **Visual upgrades:** sprite/3D Supra, motion blur, tyre-temp glow, racing-line
  overlay coloured by grip usage, brake-disc glow.
- **Telemetry export** — log laps to CSV; plot best-lap traces; compare brains.
- **Neural-net visualiser** — draw the live network graph with edge activations.
- **Tune-the-car UI** — sliders for boost, weight distribution, diff, tyres, and
  watch how the *learned* driving style changes.

### Scale & training infra
- ✅ **Done — Multiprocessing rollouts + checkpoint/resume.** PPO gathers
  experience across many envs in parallel worker processes (`--workers`), and
  checkpoints (weights + obs-normaliser + curriculum stage) save/resume
  automatically.
- **GPU training.** The net is small and CPU-bound on the env; a GPU helps most
  if the network grows or batches get large.
- **Vectorised batch sim** (numpy/torch) to run thousands of cars in parallel.
- **Hyperparameter search** over network size, reward shaping (`PPOSpec`).
- **Track generalisation** — evaluate each brain on several random tracks per
  generation so it learns *driving*, not one circuit.

---

## Tuning cheatsheet (`supra/config.py`)

- More power: raise `torque_curve` values or `max_boost_bar`.
- More tail-happy / drifty: lower `tyre_mu`, raise `brake_bias_front`, move
  `front_weight_frac` down (more rear weight).
- Faster GA learning signal: raise `EvoSpec.population`, tune
  `mutation_rate` / `mutation_scale`.
- Harder track: raise `radius_jitter`, lower `min_corner_radius`, narrow
  `half_width`.
- **PPO tuning (`PPOSpec`):** more parallelism with `n_envs` / `n_workers`;
  a longer `horizon` or more `epochs` for a stronger update; `progress_weight`
  is the speed objective, `crash_penalty` / `jerk_weight` shape how safe vs.
  aggressive the line is; `max_episode_seconds` sets how much track each episode
  sees; `curve_preview_dists` (in `SensorSpec`) is how far ahead it plans.

Built as a foundation to grow toward superhuman mini-Supras. Have fun. 🏁
