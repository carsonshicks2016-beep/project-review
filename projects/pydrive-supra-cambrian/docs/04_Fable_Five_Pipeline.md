# The Fable Five Pipeline

> **Edition boundary:** Fable Five has identity-isolated Mazda 787B (`fable-v1`)
> and Porsche 919 Evo (`fable-v2`) editions. The Fable 919 remains a legacy
> approximation: it may be trained, replayed, and inspected, but it is never
> faithful-v2 or certification evidence. The evidence-gated authority program is
> separately documented in [`faithful-v2`](07_Faithful_919_Record_Program.md).

Fable Five (`supra/fable5.py`) is a dedicated, edition-scoped pipeline engineered to train the Mazda 787B and the legacy-approximation Porsche 919 Evo on the Nordschleife. It replaces the traditional "drive as far as you can" reward with a highly structured, physics-driven pace curriculum while keeping each car's manifests, checkpoints, observation layout, drivetrain, diagnostics, and benchmark isolated.

## The Physics-True Speed Envelope

Instead of learning from scratch, the observation space for the Fable Five pipeline features a **Pace Block**. A theoretical physics-true speed envelope is computed from the real 787B car model across the entire track. The network sees the correct speed for its current position plus 6 points down the road ahead, allowing it to anticipate braking zones based on the physics floor rather than trial-and-error alone. 

- **Observation Space**: Layout `fable-v1`, 68-dimensional (includes the Pace Block). Hybrid-ledger editions (Porsche 919 Evo) ride `fable-v2`, 70-dimensional — battery SOC and signed MGU power appended after the Pace Block (`supra.fable5.obs_layout_for`).
- **Reward**: Pace relative to the theoretical physics envelope.

## The 5 Gated Stages

The pipeline runs across five gated stages that automatically advance unattended:

1. **Foundation**: Learns the layout at a baseline pace.
2. **Flow**: Connects sectors with smooth inputs.
3. **Finish**: A segment stage demanding a clean lap closure. The agent is required to survive the track to bank a full lap. To learn this, it spawns 30% of its episodes in the final 70–98% of the lap (lap-closure drills).
4. **Fast**: Adaptive segment training that pushes the speed envelope towards 0.96x physics capability.
5. **Frontier**: Adaptive segments that push beyond the envelope (1.15x), searching for racing-line gains that exceed the theoretical centerline floor.

```bash
# Run the entire gated ladder unattended
python3 run.py --fable 30000 --fable-stage auto

# Watch the best Fable Five checkpoint
python3 run.py --watch-fable
```

## The Pit Wall Supervisor

The Fable Five pipeline is overseen by an automated Pit Wall Supervisor (`fable5.PitWall`). It watches the rolling windows of evaluation data and makes high-level decisions.

- **Rollbacks & Reseeds**: If the agent's performance collapses (e.g., forgets how to drive a sector), the Pit Wall immediately rolls back the weights to the last known "best" state.
- **Consolidate Lever**: If reseeds stop working in a dry spell, the Pit Wall halves the learning rate and entropy to stabilize learning.
- **Weak-Sector Memory**: Eval failures persist as a heatmap. The next run deliberately seeds exploring-starts just BEFORE the problem zones to rehearse and fix the weakness.

The Pit Wall is the learning controller inside one trainer. Headless jobs are
also wrapped by `tools/supervise_fable5.py`, which provides the operational
layer: one-owner locking, durable logs, a sleep-prevention assertion, checkpoint
heartbeat/stall recovery, disk protection, bounded crash restart, and whole
process-group cleanup.

```bash
python3 tools/supervise_fable5.py --name fable5_ring -- \
  python3 run.py --fable 30000 --fable-stage auto --out fable5_ring.pt
```

Every destination stage evaluates its incoming policy at update zero. A stage
can therefore gate immediately without risking an optimizer step, while stale
checkpoints are re-certified under the current code, track asset, full-body
boundary rules, and evaluation protocol.

## Validation & Benchmarks

To ensure the model is superhuman, it is evaluated against historical benchmarks:
- **Current reachable centerline envelope**: ~8:42.22
- **Stefan Bellof '83**: 6:11.13 (The "Superhuman" benchmark)
- **Porsche 919 Evo**: 5:19.55

The dashboard's **Pit Wall tab** displays these benchmarks visually against the current best AI lap.

The centerline envelope is a reachable reference profile, not proof that the
current simulator can reach the 6:11 benchmark. Racing-line gains may beat the
centerline reference, but autonomous training cannot guarantee constant gains;
the safeguards guarantee that certified best evidence is not overwritten by a
regression and that stalls/failures become visible and recoverable.

### Validation gates

```bash
python3 tools/validate_fable5.py
python3 tools/validate_fable5_auto.py
python3 tools/validate_fable5_safeguards.py
```

These cover physical envelope reachability, stage transitions, checkpoint and
policy-hash provenance, full-body track validity, raw observation normalization,
transactional KL rejection, baseline gating, global champion protection,
duplicate training exclusion, and supervisor process-tree cleanup.

### Brain Lab Diagnostics
The `Brain Lab` (`supra/fable5_diag.py`) runs a checkpoint through an envelope-aware battery: 2 flying laps, 16 sector starts, and 5 lap-closure drop-ins. It generates dense forensic reports identifying exactly where and why the car crashes (e.g., traction-break, overspeed, ran-wide).

## Baseline: the genetic algorithm vs PPO (same task, different learner)

`supra/fable_ga.py` is a deliberately rudimentary yardstick for the PPO pipeline:
a **genetic algorithm** (elitism + tournament selection + uniform crossover +
Gaussian mutation with annealing sigma) evolving a population of NumPy-MLP
genomes. The point is a *fair fight* — it reuses Fable Five's frozen machinery so
the only difference from `--fable` is the learner, not the problem:

| Held identical to PPO | How |
| --- | --- |
| Car / track | 787B on the real-elevation Nordschleife (`FABLE_CAR` / `RING_TRACK`) |
| Observation | drives the same `FableEnv` → frozen `fable-v1` obs (68 dims) |
| Control | 3 actions `[steer, long, gear-offset]`; gear is the same ±2 RaceBox offset |
| Reward | both optimisers maximise the identical `FableReward` return |
| Network capacity | default hidden `(128, 128)` = the PPO actor's |
| Obs standardiser | a frozen copy of PPO's `RunningNorm` (same clip/formula) |
| Yardstick | the champion is scored by the identical `FableEvaluator` lap eval |

Because the eval is identical, the GA prints `[eval-ga]` lines that are directly
comparable to `[eval-fable]` (same `metric`, `lap_time`, `clean_sectors`,
`vs_bellof`). The GA writes its own checkpoint `ga_787b_ring.npz` and dashboard
snapshot `fable_ga_eval_latest.json`; it never touches Fable Five state.

```bash
# Headless (strongest): evolve the GA baseline, fair equal-time vs a PPO run
python3 run.py --fable-ga 300 --fable-ga-stage frontier \
  --pop 96 --workers 8 --out ga_787b_ring.npz          # + --fable-ga-minutes 120

# Live: watch the whole 787B population launch and evolve on the Ring
python3 run.py --fable-ga 300 --live --pop 28

# Resume, then watch the champion drive
python3 run.py --fable-ga 300 --resume ga_787b_ring.npz --out ga_787b_ring.npz
python3 run.py --watch-fable-ga --checkpoint ga_787b_ring.npz
```

Fitness is a bounded-horizon `FableReward` return averaged over a fixed
deterministic start set (the analogue of PPO's finite rollouts); the expensive
full-lap `[eval-ga]` runs every `--ga-eval-every` generations (default 5). The
comparison is honest about its expected outcome: evolving ~26k weights with a
population is far slower than the policy gradient, so the GA is the *rudimentary*
model — the interesting question is how close it can get on identical footing.

## Versioned five-speed drivetrain

The current Mazda training identity is `mazda787b-5spd-ring-v1`. It retains the
frozen `fable-v1` 68-observation/3-action contract, but replaces the former
six-speed car with five optimized forward ratios and a torque-aware sequential
RaceBox. Checkpoints store this identity; six-speed checkpoints may seed a new
foundation run only through the explicit migration path, which preserves the
steering/longitudinal policy and resets the gear action plus gear normalization.
They are never accepted as certified five-speed gate evidence.

The ratio selection is reproducible with `tools/optimize_787b_gearing.py`; the
committed result is `supra/data/mazda787b_5spd_ring_v1.json`. Run
`python3 tools/validate_787b_drivetrain.py` before five-speed training. Ratios
remain frozen throughout a lineage; changing them requires a new drivetrain
version and complete recertification.

Migration rollout (only after the six-speed supervisor has exited):

```bash
python3 tools/archive_fable5_lineage.py --prefix 12345678 \
  --label mazda787b-6spd-legacy
python3 tools/validate_787b_drivetrain.py
python3 tools/supervise_fable5.py --name fable5-ring-5spd-v1 \
  --max-restarts 12 --stall-minutes 30 --min-free-gb 10 -- \
  python3 run.py --fable 49901 --fable-stage auto --car mazda787b \
  --workers 64 --pop 64 --resume fable5_ring_best.pt \
  --out fable5_ring_5spd_v1.pt
```

The resume is a warm-start only: drivetrain mismatch invokes the migration
path, resets updates/gear semantics, and begins five-speed certification at
foundation. Never run this command concurrently with the legacy 64-worker job.
