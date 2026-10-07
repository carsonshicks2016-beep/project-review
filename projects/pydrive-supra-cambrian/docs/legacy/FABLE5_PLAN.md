# Fable Five — PPO Nürburgring Training Pipeline

> A ground-up redesign of the 787B Nordschleife program, aimed at **superhuman
> clean laps**: beat the fastest lap a human has ever driven in comparable
> machinery (Stefan Bellof, Porsche 956, 6:11.13 = 371.13 s), then chase the
> physics-optimal lap of this simulator's car+track.

**Module:** `supra/fable5.py` · **CLI:** `run.py --fable` · **Dashboard:** "Fable Five" tab
**Checkpoints:** `fable5_ring_<stage>.pt` (+`_best`), champion promoted to `fable5_ring_best.pt`
**Manifest:** `fable5_ring_pipeline.json` · **Eval line:** `[eval-fable] key=val …`

---

## 1. Why the old pipeline capped out (diagnosis)

The `ring_race` pipeline (survive→fast→attack) reached 71% clean lap progress at a
mean of ~27 m/s and then entered a repair spiral. Three structural ceilings:

1. **A fixed lateral-g assumption.** Its safe-speed cap used `lat_accel_limit`
   ≤ 12 m/s². The 787B (μ=1.8 slicks + ClA 3.5 downforce) pulls **25–30 m/s²**
   once aero loads up. The overspeed/late-brake penalties actively *punished*
   driving at the car's real limit — the reward taught a fast road car, not a
   Group C prototype. That alone caps pace at roughly half of what physics allows.
2. **Unreachable first milestone.** The attack eval's line run was budgeted at
   540 s; a clean lap literally could not register below a 38.6 m/s average, so
   `best_clean_lap` stayed `null` no matter what and keep-best had no lap-time
   signal to climb.
3. **Repair-run forgetting.** Fixing one corner in a separate short run degraded
   the rest (75% terminal rate in the last repair eval). The fix has to live
   *inside* the run (rehearse weak sectors continuously), not outside it.

## 2. Is PPO the right algorithm?

**Kept — deliberately.** The reward here is dense (progress every step), the
horizon is effectively local (pace now, brake in the next ~5 s), and this
codebase has a proven, restart-hardened PPO with keep-best, plateau reseed, and
subprocess rollouts. Off-policy alternatives (SAC/TD3) buy sample efficiency at
the cost of new failure modes on CPU; the previous ceiling was **not** the
optimizer — it was the reward's physics model, the observation, and the
curriculum. Those are what Fable Five replaces. (If PPO plateaus above the
human benchmark but below the theoretical lap, the manifest's plateau data will
say so, and an off-policy fine-tuner can be bolted onto the same env.)

## 3. The core new machinery

### 3.1 Physics-true speed envelope (the centerpiece)
`compute_speed_envelope(track, car)` runs a classic three-pass lap-time-sim
solver over the real track geometry using the **actual car model constants**
(μ with load sensitivity, ClA downforce, CdA drag, engine power from the torque
curve, RWD traction fraction, per-point grade from the DEM):

- cornering cap per point: solve `v²k = μ(v)·(g + q·v²)` (downforce-aware,
  fixed-point iterated; `q = ½ρ·ClA/m`),
- forward pass: traction/power-limited acceleration through the friction
  ellipse, minus drag and grade,
- backward pass: braking (grip + drag + grade) through the friction ellipse.

This yields `v_ref(s)` — the fastest speed profile the sim's own physics can
hold on the centerline — and the **theoretical lap time**, the superhuman
yardstick stored in the manifest. Utilization factors (`lat_util`, `long_util`)
absorb model mismatch (relaxation lengths, transients) and are the calibration
dials.

The envelope is used three ways:
1. **Observation** — a new pace block (see 3.2): the policy *sees* the correct
   speed for here and for 6 points down the road. Braking points become
   pattern-matching instead of trial-and-error.
2. **Reward** — pace is rewarded as `v/v_target(s)` (envelope-relative), so a
   hairpin taken perfectly earns like a straight taken perfectly. Overspeed
   beyond the envelope is softly penalized early and nearly ignored late
   (racing lines legitimately beat the centerline envelope).
3. **Eval** — lap-time budget and benchmark ratios derive from it.

### 3.2 Observation: `fable-v1` layout (58 → 66 sensors, +2 mode = 68)
Appends 8 dims to the sensor vector (old indices untouched):
`v_ref` at {0, 25, 50, 100, 175, 275, 400} m ahead (normalized /90) + the
current speed-vs-envelope ratio. Implemented inside `SensorSuite` (gated by
`SensorSpec.pace_block`), so training, watching, and the live viewer all build
the identical observation. Checkpoints record the layout; the loader
reconstructs it.

### 3.3 Warm-start transplant
`ring_787b_best.pt` (71% clean progress) is too valuable to discard. A
transplant loader copies its trunk weights column-mapped into the 68-dim net
(58 sensor cols + 2 mode cols land in their new positions; the 8 pace columns
start near zero) plus the normalizer stats. Foundation starts from a policy
that already drives 15 km of the Ring clean.

### 3.4 Weakest-sector replay (in-run repair)
Each env tracks which of 24 track sectors its episodes die in and biases its
exploring starts toward those sectors (decaying weights). The trouble corner
gets rehearsed *within* the run while the rest of the lap keeps training — no
more separate repair runs that forget the other 20 km.

### 3.5 Rehearse at race pace
Exploring starts spawn at 0.75–1.05× the *envelope* speed for that point
(stage-scaled), so the policy practices corner entries at the speeds it will
actually carry — the old pipeline capped spawn speeds at 38 m/s.

## 4. The five stages

| stage | goal | envelope scale | gate to next |
|---|---|---|---|
| **1 FOUNDATION** | drive clean everywhere at moderate pace (transplanted from ring best) | 0.78 | ≥14/16 clean eval sectors, terminal ≤ 12.5% |
| **2 FLOW** | chain the whole lap clean at speed, full track width | 0.86 | 16/16 clean chain, pace ratio ≥ 0.55 |
| **3 FINISH** | bank the first clean lap (line starts, 1000 s budget) | 0.90 | a clean lap exists |
| **4 FAST** | compress lap time (pace-dominant reward) | 0.96 | clean lap ≤ 7:30 |
| **5 FRONTIER** | superhuman: envelope at 1.0, overspeed shaping ~off, pure pace | 1.00 | plateau vs benchmarks |

Stage-scheduled: align/edge/slip shaping only in early stages; entropy, LR,
log_std, patience, reseed budget all tighten toward frontier. `--fable-stage
auto` runs the ladder with gates; each stage warm-starts from the previous
stage's `_best` and promotes to `fable5_ring_best.pt`.

**Benchmarks (manifest + dashboard):** theoretical envelope lap ·
Bellof 956 371.13 s (superhuman line) · Porsche 919 Evo 319.55 s (outright
record). A clean lap is only *valid* with zero off-track time (real Ring rules).

## 5. Eval protocol (per `eval_every`)
16 deterministic sector runs (70 s each, envelope-paced starts) → clean rate,
clean chain, pace ratio; **one line run with a generous budget**
(`max(900 s, 2.4× theoretical)`) so a slow-but-clean lap can register from day
one — fixing ceiling #2. Metric: clean-driving score early; `100000 / lap_time`
once laps exist. Keep-best, plateau reseed, and early stop ride on it as usual.

## 6. Files touched
- `supra/fable5.py` — envelope, spec/reward, env, evaluator, train/watch (NEW)
- `supra/sensors.py` + `supra/config.py` — pace block (opt-in, default off)
- `supra/ppo_env.py` — plumb pace spec fields into SensorSpec
- `supra/ppo.py` — save/load layout awareness (`fable-v1`)
- `run.py` — `--fable`, `--fable-stage`, `--fable-scale`, `--watch-fable`
- `command-center/server.py` — fable actions, `[eval-fable]` parsing, `/api/fable-five`
- `command-center/static/{index.html,app.js,style.css}` — "Fable Five" tab
- `tools/validate_fable5.py` — envelope sanity + env + train smoke gates
