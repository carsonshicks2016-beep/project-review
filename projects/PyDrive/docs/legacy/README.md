# Supra Drift

A from-scratch neuro-evolution / reinforcement-learning **drifting & grip-racing
simulator** in Python. Custom four-wheel vehicle dynamics, procedural tracks, a
genetic-algorithm path and a PyTorch PPO path (race **and** drift), live engine
audio, an in-app watch viewer, and a self-contained web dashboard.

Top-down 2D, real vehicle physics. Trains a Toyota Supra (with RX-7 and R34
Skyline presets). Audio synthesises live. This archived overview predates the
clean-room Fable Five Observatory now served by Command Center at
`/observatory/`; the removed browser viewer is not available as a fallback.

## Status

| Area | What | State |
|------|------|-------|
| Sim | Config + four-wheel physics + procedural tracks + drivable 2D app | ✅ |
| Hills + jumps | 2.5D slope physics + emergent airborne mode — tracks have real elevation; crest too fast and you FLY | ✅ |
| Sensors | 9 beams + 25 proprio + look-ahead curvature + 18-dim hill/air block → 58-dim obs | ✅ |
| GA | NumPy MLP brains, elitism + tournament + crossover + mutation (race) | ✅ |
| PPO race | PyTorch actor-critic, GAE, curriculum, deterministic eval + keep-best | ✅ completes laps |
| PPO drift | Slip-angle reward, anti-spin shaping, completion-gated curriculum | ✅ drifts; ⏳ hardest touges still spin out |
| Hybrid | One goal-conditioned race/drift policy (infra ready: obs carries the mode one-hot) | ⏳ next |
| Hills retrain | New checkpoint generation on hilly tracks (pre-hills checkpoints refuse to load, loudly) | ⏳ next |

The 60-dim observation is **mode-conditioned** (`[race, drift]` one-hot), so one
network architecture trains race, drift, and a future goal-conditioned hybrid —
just swap the reward and flip the flag.

## Install

```bash
pip3 install -r requirements.txt    # numpy + pygame + torch + flask (+ optional sounddevice)
```

This Mac uses `python3` / `pip3` (not `python` / `pip`). Torch runs CPU-only
(`DEVICE = cpu` in `supra/ppo.py`); no GPU required.

## Command Center (web dashboard)

The one-stop control panel for everything — drive, train (GA / PPO race / PPO
drift), watch, manage checkpoints, browse tracks. All-Python (Flask), no Node.

```bash
python3 command-center/server.py        # then open http://localhost:8770
```

Or double-click **`~/Desktop/Supra Command Center.command`** (or
`command-center/SUPRA.command` in-repo). It launches `run.py` subprocesses,
streams logs + parsed training metrics over SSE (live charts), and re-attaches to
running tasks on refresh. Training-code edits apply to new runs automatically (it
spawns `run.py` fresh); only `server.py` changes need a dashboard restart.

> The Node `project-dashboard/` is a *separate, general* multi-project homepage
> (port 8000, `~/Desktop/HOMEPAGE.command`) — not part of Supra Drift. Use the
> Command Center above for this project.

## Drive it yourself

```bash
python3 run.py --drive --track club             # a named track
python3 run.py --drive --track akina --car rx7  # tail-happy RX-7 on a touge
python3 run.py --drive --gen technical --difficulty 0.85 --length 1400
```

**Controls:** arrows / WASD to drive · Space = handbrake · Shift = clutch ·
T = toggle auto gearbox · Q/E = manual down/up shift · **C = camera mode** ·
**V = broadcast director** · **1-5 = direct director shots** · **P = replay** ·
**I = complete AI Vision overlay** · **B = sensor beams** ·
**G = day/dusk/night** · **H = weather** · **Y = visual wet surface** ·
**O = HQ post-processing** ·
**F3 = performance diagnostics** · **F11 = fullscreen** ·
**M = mute audio** ·
R = reset · Esc = quit.

### The 2D viewer: V2 (projected 2.5D) is the default

Drive and watch modes open **Viewer V2** (`supra/viewer2.py`): one faux-3D
projector for the whole world — the road renders with **real elevation**
(crests bulge, dips sink, banking tilts the edges), a heading-up chase camera
rotates the world around the car, and the car's shadow stays glued to the road
while the body lifts on jumps. Kerbs, checkered start gate, trackside props,
tyre smoke, skids, night headlights, and a Fable-styled amber HUD (rpm arc,
clean-lap board, drift bar, elevation-tinted minimap). The classic V1 viewer
(`supra/app.py`) is untouched and always available: `--classic` on any
drive/watch command (or `SUPRA_CLASSIC_VIEWER=1`). Multi-car race modes
automatically use V1.

The real Nordschleife asset also drives a track-specific visual layer: named
section signs and callouts, T13 pit structures, concrete Karussell surfaces,
Brünnchen/Pflanzgarten spectator camps, Döttinger Höhe gantries, road graffiti,
camera towers, landmark structures, and low-elevation forest mist.

`H` cycles clear, overcast, rain, storm, and forest mist. Weather remains
visual-only, but it progressively changes rendered wetness and spray. Session
patina persists across resets: rubber builds on the racing surface, off-track
tracks remain in the terrain, and the car accumulates grime. `O` enables the
more expensive bloom, high-speed temporal persistence, and impact color split.

Trackside scenes are also alive: spectator groups fire independently timed
camera flashes, stationary Supra, LR4, RX-7, and F-150 faux-3D models run
asynchronous hazard lights, and billboard or gantry displays rotate messages
with scan lines and status beacons. Parked models are individual world-space
objects, depth-sorted and distance-fogged alongside trees for correct parallax
and occlusion.

Sparse wind turbines now occupy distant high/open ground. Their towers and
three-blade rotors are world-anchored, slowly animated, fog-aware, and naturally
move with less parallax than the nearby forest. Atmospheric haze uses a broad
smoothstep feather, ordered alpha dithering, and premultiplied layer composition
to prevent horizontal shading bands in scaled or fullscreen presentation.

Viewer V2 interpolates between consecutive 120 Hz physics states, so camera,
car, and track movement remain continuous even when a render frame lands
between simulation steps. Presentation FPS follows the active macOS display
refresh rate (120 Hz on the built-in ProMotion panel); `SUPRA_RENDER_FPS=144`
can override it for a 144 Hz external display. The window uses scaled VSync and
F11 toggles fullscreen without increasing the internal 1480×820 render load.
F3 shows delivered FPS, frame time, 1% low FPS, catch-up steps, and current FX
quality. An automatic governor protects motion cadence by reducing film grain
and edge-speed decoration before it ever reduces world or vehicle geometry.

Audio-connected presentation is routed through one render-only coordinator. It
tracks engine phase and harmonics, throttle load, resonance bands, shifts,
downshift backfires, limiter gates, and camera-relative radial velocity without
writing to the vehicle or modifying the audio stream. The shared signal now
drives harmonic tailpipe breathing, gear-change pressure rings, synchronized
backfire/limiter flashes, resonance-only body and camera vibration, and a brief
RPM waveform trace in the HUD after every shift. Fixed trackside and apex shots
also visualize Doppler motion: approach wavefronts compress, the receding wake
stretches, and closest passage produces one synchronized pressure ring and
camera punch. The T13 pit apron is staffed by animated marshals, lollipop and
pit-board operators, tyre carriers, and crouched mechanics with role-specific
signals, tools, walking cycles, and independently timed activity.

## Tracks

One difficulty-parameterised generator spans four archetypes; corner density
scales with length so long tracks stay technical.

```bash
python3 run.py --drive --track speedbowl     # long straights — wind it out
python3 run.py --drive --track tech          # tight technical, multi-apex
python3 run.py --drive --gen gp --difficulty 0.2 --length 800
```

Named set: `club national coast sprint tech oval2 akina pass ridge speedbowl
superspeed` (+ legacy `oval random touge`). `--gen` archetypes: `gp` (flowing) ·
`technical` (hairpins) · `speedway` (fast) · `touge` (switchbacks). The PPO
generalist curriculum dials difficulty up as the agent improves.

**Every track has real elevation now** (style-shaped, seed-deterministic):
climbs pull, descents push, crests lighten the car — and a crest taken fast
enough launches it (`ridge` is the canonical jump track: two authored crests,
~103 and ~115 km/h launch speeds). `--flat` kills elevation everywhere (the
pre-hills feel, A/B lever); `--hill-scale 0..1.5` dials amplitude.

## Fable Five — the superhuman Nürburgring pipeline

The dedicated 787B Nordschleife program (`supra/fable5.py`, design in
`FABLE5_PLAN.md`): a physics-true **speed envelope** computed from the real car
model feeds the observation (a pace block: the correct speed here + 6 points
down the road), the reward (pace relative to the envelope), and the eval
(benchmarks: theoretical lap · Bellof '83 6:11.13 · 919 Evo 5:19.55). Five
gated stages — foundation → flow → finish → fast → frontier — with
weakest-sector replay, race-pace exploring starts, and clean-lap rules (any
off-track invalidates the lap).

```bash
python3 run.py --fable 6000 --fable-stage foundation   # stage 1 (auto-transplants ring_787b_best.pt)
python3 run.py --fable 30000 --fable-stage auto        # the whole gated ladder
python3 run.py --watch-fable                           # watch fable5_ring_best.pt
```

Or drive it from the Command Center's **Fable Five** tab (stage board, lap
board vs the human benchmarks, watch/diagnose). Checkpoints use the `fable-v1`
obs layout (68-dim, pace block) — old checkpoints still load everywhere else.
Validation gates: `python3 tools/validate_fable5.py`.

## GA — the first AI (race)

A population of NumPy MLP "brains" evolves to drive the track. Fitness =
centreline distance; breeding = elitism + tournament + uniform crossover +
Gaussian mutation. GA has inherent keep-best via elitism (the saved champion is
always the best ever seen), so longer is never worse.

```bash
python3 run.py --train 200 --track akina --out akina_ga.npz   # specialist
python3 run.py --train 100 --live                              # WATCH the swarm evolve
python3 run.py --watch --checkpoint akina_ga.npz              # watch the champion
```

## PPO — race & drift (the strong AI)

PyTorch actor-critic PPO: GAE, clipped surrogate, value + entropy losses,
in-process vectorised envs, and a curriculum. The actor has a forward-throttle
init bias (so the untrained policy moves), and `log_std` is clamped to `[-2.2, 0]`
(stops entropy from exploding in long runs).

```bash
# RACE
python3 run.py --ppo 600                                   # generalist (curriculum)
python3 run.py --ppo 800 --track akina --out akina_race.pt # specialist
python3 run.py --watch-ppo --checkpoint ppo_race.pt        # watch

# DRIFT
python3 run.py --drift 1500                                # generalist drift
python3 run.py --drift 2000 --track national --out national_drift.pt
python3 run.py --watch-drift --checkpoint ppo_drift.pt     # watch (auto picks the trained track)

# continue / branch any run
python3 run.py --drift 1000 --resume some.pt --out some_v2.pt

# LIVE windows (watch it train; S = save / L = load via file dialog, Space = pause)
python3 run.py --ppo 100000 --live
python3 run.py --drift 100000 --live --track national
```

Flags: `--car {supra,rx7,skyline,lr4,f150}`, `--track NAME` (specialist) or
`random` (generalist, default), `--out NAME` (named save), `--resume PATH`,
`--live`, `--no-audio`, `--seed N`, `--pop N`, `--flat` (no elevation),
`--hill-scale X` (0..1.5).

### Judge drift by the `[eval]` line

Training prints two kinds of numbers. The rolling `drift` / `laps` in each log
line are measured under exploration noise + exploring-starts and **overstate**
reality. The trustworthy number is the periodic **`[eval]`** line: a
**deterministic, full-episode** run from fixed points around the whole loop
(`act_mean`, no noise) — exactly what you see when watching. Keep-best (`_best.pt`)
and early-stop key off this. Composite (drift) = `drift × (0.3 + 0.7·lap)`, so a
"suicide drifter" (big angle, spins out, no progress) and a "grippy racer"
(drift≈0) both score ~0; only **drift-while-completing** wins.

## Engine audio (synthesised live, no samples)

Real-time additive synthesis driven by telemetry (`supra/sound.py`, via
`sounddevice`): raspy inline-six harmonics that harden with load, a turbo whine
that spools with boost, **turbo flutter** ("stu-tu-tu-tu") on lift, tyre squeal
scaled by slide, and exhaust crackle on downshifts/overrun. Degrades gracefully —
no audio device just means silence. Disable with `--no-audio`, mute live with **M**.

## What the agent sees

- **Vision** — 9 raycast beams (±100°, 70 m) to the track walls.
- **Proprioception** — 25 signals: body velocities, yaw rate, g-forces, chassis
  slip, per-wheel load / grip-usage / slip ratio, rpm, gear, boost, steering, and
  where it sits on track.
- **Look-ahead** — signed centreline curvature at 6 distances down the road (8–90 m).
- **Hills & air (18)** — road grade/bank under the body, actual pitch, vertical
  speed, height above the road, an airborne flag, plus look-ahead grade AND
  look-ahead crest curvature at the same 6 distances — with its own speed, the
  net can predict a takeoff before it happens and time the landing.

That's a 58-value observation, plus the 2-way mode one-hot → the 60-dim vector
the GA (58 only) and PPO policies consume. Drive with the dashboard up (Tab) to
watch it live as a bar field.

## Architecture

```
run.py                CLI entry point — arg parsing + dispatch to all modes
command-center/       Self-contained web dashboard (Flask + vanilla JS), port 8770
observatory/          Current Fable Five 3D Brain Observatory (local Three.js/Vite client)
supra/
  config.py           ALL dataclasses: CarSpec(+presets), SimSpec, SensorSpec,
                      EvoSpec, PPOSpec, RaceReward, DriftReward
  physics.py          4-wheel dynamics: Pacejka tyres, weight transfer, drivetrain
  track.py            Catmull-Rom circuits, raycasts, look-ahead, generators, named tracks
  sensors.py          SensorSuite -> the 40-dim observation
  brain.py            NumPy MLP genome (GA)
  evolution.py        the GA: elitism, tournament, crossover, mutation, checkpoints
  agent.py            CarAgent: a brain drives a car, with fitness + crash detection
  ppo_env.py          SupraEnv: Gym-style, mode-conditioned, reward + RunningNorm
  ppo.py              PyTorch actor-critic PPO: GAE, clipped objective, curriculum,
                      deterministic full-episode eval + keep-best + early-stop
  drift.py            DriftScorer — the drift reward (angle×speed, anti-spin, sustain, chains)
  app.py              PyGame drive/watch loop, dashboard, audio, beams, predicted path
  app_ga.py           live population-evolution view (--train --live)
  app_ppo.py          live PPO training view (--ppo/--drift --live, S/L save/load)
  carart.py           procedural top-down car art (Supra/RX-7/Skyline)
  fx.py               particle FX (backfire flames; smoke/dust/sparks available)
  aiviz.py            AI-brain viz: PolicyAgent adapter + predicted-path drawing
  dashboard.py        in-viewer telemetry panel + raycast beam drawing
  sound.py            EngineAudio — real-time engine/turbo/BOV/tyre/backfire synthesis
```

See `HANDOFF.md` for the deep reference: physics model details, training systems,
the development journey, the checkpoint inventory, and gotchas.

### The physics model (the part that matters)

- Per-corner slip angle + slip ratio at four independent contact patches.
- **Combined-slip Pacejka** via friction-ellipse normalisation — one shared grip
  budget, so power-on wheelspin bleeds lateral grip → genuine throttle-on
  oversteer (not a scripted effect).
- **Tyre relaxation length**: lateral forces build over distance → real transient
  feel through drift transitions (no instantaneous grip).
- Weight transfer (longitudinal + roll-stiffness-distributed lateral) plus
  speed-dependent **downforce**, settled through a suspension lag; load-sensitive
  peak friction.
- Sub-stepped drivetrain: engine flywheel → clutch → 6-speed gearbox → limited-slip
  diff → driven wheels, with sequential twin-turbo spool. Free-rolling wheels
  handled kinematically for low-speed stability.
- Aero drag opposes the **full velocity vector** (sideways slides scrub speed),
  plus rolling resistance, rate-limited steering, and an off-track grip multiplier.
- **2.5D road plane**: the track feeds grade/bank/vertical-curvature per step;
  slope gravity is a real body force projected onto the car's axes (slide
  sideways down a climb and the pull is on your *side*), normal load scales
  with slope and with v² over crests/dips, and pitch/roll come out as telemetry.
- **Emergent jumps**: when a crest's v² term fully unloads the car, it takes off
  — no scripted jump zones. Flight is ballistic (planar momentum conserved
  minus drag, yaw rate frozen, engine free-revs), landing is a suspension
  impact whose grip cost falls out of load-sensitive friction. `--flat`
  restores byte-identical pre-hills physics.

Validated: 0–100 km/h ≈ 5.3 s with realistic turbo lag; lifting vs flooring the
throttle mid-corner swings slip angle from ~10° to ~50° (real combined-slip
oversteer); stable under sustained drift and off-track without numerical blow-up.
Hills validated analytically (`tools/validate_hills.py`, `validate_jumps.py`):
slope pull = −g·grade exact, coast-down energy books to 0.6%, takeoff threshold
within 2% of √(g/|vcurv|), flight range 1.2% off the parabola, flat-ground
byte-identity enforced by `tools/regression_baseline.py` at every stage.
