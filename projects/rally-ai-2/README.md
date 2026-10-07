# RallyAI

A reinforcement-learning agent teaches itself to drive a rally car flat-out,
rendered as a PlayStation-1 rally game.

The authoritative statement of what this project is lives in
[docs/north-star.md](docs/north-star.md). Where this README and the North Star
disagree, the North Star wins — but this README describes **what the code
actually does today**, and it is a bug for it to describe anything else.

**Picking this up to work on it?** The execution plan — 11 phases, 40 work
items, in dependency order with the measurement each one has to produce — lives
as data at
[packages/viewer/public/dashboard/roadmap-status.json](packages/viewer/public/dashboard/roadmap-status.json)
and is rendered by the dashboard's ROADMAP panel. It refers to a `docs/roadmap.md`
prose companion that was never written; the JSON is the source of truth until it
is. Start at the lowest incomplete item.

## Status

Early. The walking skeleton is closed and the first training run has been made
and measured. **It did not produce a driver** — see "The first training run".

| Piece | State |
|---|---|
| Repo, package layout, venv | working |
| The three contracts (stage / replay / metrics) | schemas written, validated, tested |
| Vendored Supra dynamics | vendored and smoke-tested |
| Track geometry (corridor, projection, raycast, look-ahead) | working |
| Stage builder + authored fixtures | working, seed-reproducible |
| Car preset (`evo_rally`) + road bridge + auto gearbox | working |
| Sensors (vision / proprio / look-ahead / terrain+air) | working, 84 dims |
| Environment (4 actions, 6 terminations, replay export) | working |
| Procedural stage generator (6 tiers, archetype vocabulary) | working |
| Reward | working — edge taper, super-linear speed, time, landings, steer/overspeed costs |
| Vectorisation | working — sync + async workers, SIGKILL revive |
| Trainer | working — PPO, KL guard, hall of fame, mastery curriculum, staged pipeline |
| Evaluation harness | working — held-out seeds, sectors, vs-optimal; `time_vs_human` still null |
| Control room | working — FastAPI job control, live agent stream, live human drive |
| Viewer | WATCH + TRAIN + dashboard — stylized low-poly scene, cameras, HUD, dust, procedural audio |

The walking skeleton is closed: `seed → stage → env → replay`, with the replay
validating against its schema. One run of 10.9M steps has been trained and
evaluated. **There is still no result to claim** — the policy finishes 30% of
held-out tier-0 stages, and the run diagnoses rather than delivers.

### Measured

Numbers here are re-derived by running the code, not remembered. Geometry cost
per environment step, on a 12-core M-series laptop, 538 m stage with 86 trees:

| Operation | Cost |
|---|---|
| `project` (hinted) | 16.5 µs |
| `raycast`, 9 beams, 60 m | 58.7 µs |
| `lookahead_*` (6 distances) | 4.0 µs |
| `collide_obstacles` | 3.1 µs |
| **geometry per step** | **90.5 µs → ~11,000 steps/s ceiling** |

For scale: v1's *entire* environment ran at ~950 steps/s with no raycasting at
all. Vision is affordable; the physics will be the bottleneck. Raycast is 65% of
the geometry budget and is the first thing to optimise if throughput bites.

**Reward balance**, integrated over full episodes with the reference pilot, six
seeds a tier. The share of the return attributable to pace — speed, throttle
commitment, the stopwatch, and the time-scaled finish bonus:

| Tier | 0 | 1 | 2 | 3 | 4 | 5 | mean |
|---|---|---|---|---|---|---|---|
| pace share | 17.3% | 15.2% | 14.1% | 13.6% | 12.7% | 12.7% | **14.3%** |

Re-derived after `pace_ref_mps` became per-stage (`finish_s / theoretical_minimum_time`).
Still inside the ~15% target band. v1's equivalent was **0.17%**, which is why v1
produced a driver that completed stages slowly. The target of ~15% is a deliberate
choice, not a derivation: it trades stage times against completion rate, and it is
checked by integrating the terms over whole episodes after every edit to `reward.py`
or anything that changes the return scale (including `pace_ref`).

Caveat worth keeping in view: the *share* is not the *gradient*. Driving the same
tier-1 stages 24% faster (37.9 s → 30.5 s) moves the return by only +1.7%,
because progress is ~77% of the return and is pace-independent by construction.
That is still ~13× v1's pace sensitivity per unit of pace, and the super-linear
speed term steepens at speeds the reference pilot never reaches — but if training
produces a cautious driver, lowering `progress` is the lever, not raising `speed`.

**Exploits are hunted, not assumed.** Two have been probed so far and both were
real. Verge-riding — farming centerline progress from the outer edge — is closed
by the edge taper: at 0.90 of the half-width it costs −382 return. And
`throttle_commit` originally read the throttle without reading the brake, so
pinning the throttle and dragging the brakes earned **+7.9% at a matched 21.7 m/s
against an honest run at 21.1** — slower, and better paid. Closed by requiring
brake ≤ 0.05. Each fix carries a negative control that reproduces the exploit on
purpose, because a green test proves nothing until it has been seen to fail.

Corner-radius round-trip fidelity (builder → file → spline refit) is within
**5.8%** worst case across radii 10–120 m and angles 45–170°.

**Environment throughput** (30 Hz control, 120 Hz physics, 4 physics steps per
action), regenerated by `python -m rallyai.train.bench` /
`python -m rallyai.train._bench_scale` on a 12-core M-series laptop.

Best quiet-ish pass (load average ≈ 8, no other training job):

| Setup | Steps/s |
|---|---|
| SyncVecEnv n=1 | **481** |
| SyncVecEnv n=2 / 4 / 8 / 16 | 486 / 500 / 473 / 460 |
| AsyncVecEnv 1 / 2 / 4 / 8 / 12 workers | 408 / 732 / 1178 / **1624** / 1745 |

**8 workers = 3.4×** sync/1 (efficiency 3.4/8 ≈ 42%). Roadmap Done asked for
≥5×; that bar is **not met** on this host. Layout probes already supported by
the API (`envs_per_worker`) do not close it — under a contended re-check with
`foundation_01` (8 training workers) also on the machine: sync/1 470, async/8
1290 (**2.8×**), async/8×2envs 1430 (**3.1×**), async/12 1323–1640 depending
on session. Pure parallel physics without Pipe scales similarly, so the limit
is Apple Silicon memory-bandwidth / core contention, not spawn or IPC.

**Waiver:** treat B2’s ≥5× criterion as waived with the measured evidence above;
keep 8-worker steps/s as the tracked regression metric. SIGKILL revive still
passes. Profiling still puts essentially all of the per-step cost in the
vendored dynamics — `_drivetrain` and `_pacejka` dominate; geometry is not in
the top seven.

Projected for a 20M-step run at the quiet-ish 8-worker number: ~11.6 h
single-process, **~3.4 h on 8 workers**.

`evo_rally` measured on a 50 m skidpad:

| Surface | µ | Max sustained | Lateral |
|---|---|---|---|
| Tarmac | 0.98 | 104 km/h | 1.71 g |
| Gravel | 0.72 | 90 km/h | 1.27 g |
| Snow | 0.36 | 59 km/h | 0.56 g |

Grip ratios track the surface-µ ratios closely (tarmac/gravel 1.35 measured vs
1.36 predicted). 0–100 km/h ≈ 4.4 s; gear-limited to 206 km/h in fifth.

The curriculum, measured with the reference pilot over 24 seeds per tier. This
is a **drivability probe, not a baseline** — the pilot is pure pursuit plus a
curvature speed limit, and the §10 human baseline means Carson on a keyboard.

| Tier | | Completed | Mean µ | Avg speed |
|---|---|---|---|---|
| 0 | Wide Gravel Intro | 24/24 | 0.72 | 93.3 km/h |
| 1 | Forest Medium | 24/24 | 0.72 | 89.2 km/h |
| 2 | Tight Gravel | 21/24 | 0.80 | 85.2 km/h |
| 3 | Mixed Surface | 20/24 | 0.70 | 77.6 km/h |
| 4 | Snow Hazard | 9/24 | 0.54 | 66.4 km/h |
| 5 | Frontier | 7/24 | 0.51 | 60.4 km/h |

Completion and speed both fall monotonically, which is the property that makes
difficulty a single dial. Zero obstacles sit inside the drivable corridor at any
tier.

**Theoretical minimum (D2)** — reference-pilot time / forward-backward bound, same
24-seed set (finished runs only). Ratio &lt; 1.0 would mean the bound is wrong; none
observed. Means sit near **1.20×** (range ≈ 1.11–1.28). Tighter than the a-priori
1.3–1.8× guess: the pilot targets 0.85 g while the bound uses tyre µ ≈ 1.05, and
√(1.05/0.85) ≈ 1.11 already accounts for most of the gap.

| Tier | Finished | Mean ratio | Min | Max |
|---|---|---|---|---|
| 0 | 24/24 | 1.182 | 1.108 | 1.240 |
| 1 | 24/24 | 1.193 | 1.124 | 1.252 |
| 2 | 21/24 | 1.217 | 1.180 | 1.243 |
| 3 | 19/24 | 1.214 | 1.154 | 1.281 |
| 4 | 12/24 | 1.209 | 1.165 | 1.275 |
| 5 | 5/24 | 1.227 | 1.192 | 1.250 |

### The first training run

`foundation_01` — the `foundation` stage weighting, tier 0, 8 workers,
**10,866,688 steps in 1.40 h at 2,163 steps/s**. Ended interrupted. Evaluated on
held-out seeds (`seed % 10 == 7`) with mean actions, never sampled:

| Tier | Completion | Clean | Mean time | vs optimal | Terminations |
|---|---|---|---|---|---|
| 0 | 30% | 5% | 24.1 s | 1.19× | finish 6, off_course 10, crash 3, spun 1 |
| 1 | 5% | 5% | 30.3 s | 1.18× | finish 1, off_course 10, crash 6, spun 3 |
| 2 | 0% | 0% | — | — | off_course 10, crash 9, spun 1 |

The curriculum never left tier 0. Rolling completion peaked near **0.67 around
1.6M steps** and then oscillated between 0 and 0.8 for the remaining 9M without
consolidating. Two measured causes, neither yet fixed:

**Three of four action dims saturated the log-std ceiling.** `log_std` starts at
−0.5 (σ 0.61) and `DEFAULT_LOG_STD_MAX` caps it at 0.0 (σ 1.0). Read off the
hall of fame:

| Checkpoint | steps | steer | throttle | brake | handbrake |
|---|---|---|---|---|---|
| furthest | 147k | −0.579 | −0.510 | −0.476 | −0.561 |
| cleanest | 2.1M | −0.723 | −0.145 | −0.153 | −0.044 |
| latest | 10.9M | −0.857 | **−0.046** | **−0.018** | **−0.024** |

Steering went the healthy direction, σ 0.61 → 0.42. Throttle, brake and
handbrake went the other way and finished flat against the cap at σ ≈ 0.98 —
sampled at near-maximum noise on bounded actions for most of the run. Episode
entropy rose 3.67 → 5.06 over the same span, which is those three dims. The
pedals never got to practise a coherent trace, and off-course is the top
termination at every tier.

**The KL guard discarded half the compute.** The guard rolls back network *and*
optimizer whenever post-update full-batch KL exceeds `1.5 × target_kl` = 0.03.
Measured `approx_kl` sits at 0.025–0.042, so **1,356 of 2,653 updates were
rejected (51.1%)** — 39.7% in the first half of the run, **62.5% in the second**.
Roughly 5.5M steps of rollout produced no learning at all.

Note also that `reward.py` gained `steer_jerk`, `steer_sat` and `overspeed`
*after* this run, so `foundation_01` is not reproducible against current HEAD.

### Known open questions

* **Physics timestep sensitivity.** The trajectory is converged at 240 Hz
  (0.33 m from a 480 Hz reference) but 120 Hz differs by 0.87 s over the
  `proving_ground` fixture. Airborne time differs too (1.29 s vs 1.03 s), so a
  large part of that is chaotic divergence through the jump rather than
  integration error — but this has not been cleanly separated. 120 Hz is kept
  because it is the rate the vendored model was tuned and validated at. Revisit
  when jumps become load-bearing.
* **Replay size.** ~1.1 KB per frame, so a 30 s run is about 1 MB. Fine now,
  worth trimming before long runs are archived.

## Layout

```
packages/sim/         Python. The world and the brain.
  rallyai/
    contracts.py      The one implementation of load/validate/hash for all three contracts
    physics/
      vendor/         Supra Ai 2 dynamics, copied verbatim — do not edit (see VENDOR.md)
    track/            Stage geometry: corridor, elevation, camber, surfaces, obstacles
    stage/            Procedural generator (seed -> stage)
    sense/            Observation assembly (the agent's senses)
    env/              Gymnasium env: obs, action, reward, termination
    train/            PPO, vectorisation, curriculum, checkpoints
    control/          FastAPI control room
packages/viewer/      TypeScript. Vite + Three.js. Renders; never simulates.
packages/shared/      JSON schemas + authored stages. One copy, both sides read it.
```

## Setup

```bash
cd packages/sim && uv venv --python 3.14 .venv && uv pip install --python .venv -e ".[dev]"
```

## Test

```bash
cd packages/sim && .venv/bin/python -m pytest tests/ -q
```

## Invariants

These are checked by tests where possible and by review where not. Violating one
is a bug even if the code works.

- **Python owns the truth.** The browser renders state it is handed and sends
  inputs. It never steps the world.
- **One tunable surface per concern.** A number that lives in two files is one
  lie waiting to happen. Handling lives in `CarSpec`, and nowhere else.
- **The agent may only use what a driver could have.** No global stage array, no
  privileged distance-to-finish oracle beyond what pace notes give a co-driver.
- **Anything that can kill the agent must be visible to the agent.** If you add
  a way to die, you add the sense that sees it coming *in the same commit*.
- **Coordinates are z-up** (x east, y north, z up) throughout the sim and all
  three contracts. The viewer converts at load; the sim never converts. See
  [packages/shared/schemas/README.md](packages/shared/schemas/README.md).
- **Every claim is measured.** "Faster", "better", "more stable" ship with the
  number and the method, benchmarked before and after in the same session.
- **Branding is presentation-only.** The personal WATCH skin uses the approved
  historic Subaru/555-era visual language; it does not change `evo_rally`
  physics or any replay contract.

## Relationship to earlier work

- **Supra Ai 2** (`~/Desktop/Python/Supra Ai 2`) — the predecessor circuit-racing
  stack. Its vehicle dynamics are vendored here. Its PPO, sensor suite and
  reward lessons are being ported, not copied: Supra is a closed-loop,
  constant-width circuit, and this is a point-to-point, variable-width stage
  with obstacles. Adapt deliberately; a straight copy will wrap arc length past
  the finish line and read the start as upcoming road.
- **RallyAI v1** (`~/RallyAI`) — a first attempt, kept as a read-only reference
  for the PS1 render calibration. Not an ancestor of this repo.
