# Nordschleife training — handoff, 2026-10-05

Written for the agent picking this up. It records how the Nordschleife went from
"visually reviewed" to "training", what broke along the way, what is fixed, and the
**open bug you are being asked to fix** (section 4). Read section 4 first if you only
want the task; read the rest before changing anything near spawn, tags or the gate.

## 1. Current state (as of ~00:05 CDT 2026-10-05)

- Course: **`73b94c56e3e67e1b4faf28ceadb4e52d2d547c0b87daa5891e44e85a6999cfa3`**
  ("Nordschleife / full lap", 20,832 m, 420 gates, rebuilt with the tag fix). It is
  `review: reviewed` with a `trainability` block in its manifest. Older Nordschleife
  courses in `.rally/courses/` (e.g. `235e91a3…`) predate the tag fix — do not train on them.
- Training job **`21cb2eeff33940d3`** — "Nordschleife · from scratch", specialist,
  10M steps, 2 workers, seed 20261005, reward `time-attack-v1`, neutral start, no parent.
  Was **running** at ~2.0M steps (~430 steps/s, ETA ~06:00). Check
  `.rally/jobs/21cb2eeff33940d3/process.log` and `episodes/*.jsonl`.
  It is learning: mean reward −21 → ~+105 by 1.9M; last 1,000 episodes average 15.4
  gates (~770 m), best 97 gates (~4.8 km).
- **The user has not yet decided** whether to stop this run to fix the spawn bug or let
  it finish. Ask before stopping it. The probe tool cannot run while it holds
  `.rally/compute.lock`.
- Commits: `2d5fd2a` (circuit import + trainability), `065a2cc` (tag fix + obs dump).
  Uncommitted: only two stray `results/episodes/episodes-editor-20261004-*.jsonl`.
- Managed player: rebuilt after `065a2cc`, `core.build_status()` → `current`.
- Python tests: 78 pass (`.venv/bin/python -m unittest discover -s tests`; there is no pytest).
  Unity compile check: `./check-compile.sh` → `COMPILES CLEAN`.
- Dashboard: `http://127.0.0.1:8765` (`./start-rally-lab.sh`). API is used directly below.

## 2. What was changed to make the circuit trainable

| Change | Where | Why |
|---|---|---|
| Distributed training starts | `Assets/Core/ML/RallyAgent.cs` `StartAtCircuitStation` (~L636) and the `spawnProfile=="distributed"` branch in `OnEpisodeBegin` (~L707) | A fixed T13 start on a 20 km lap only ever teaches the first km. Training (not evaluation/viewer) draws a station uniformly in `[circuitStart, circuitEnd−300]` from a per-worker RNG seeded via `LabRuntime.NextSeed()`. |
| Distributed episodes never ranked, but not `labInvalid` | `distributedStart` flag in `RallyAgent.cs`; `record.valid` excludes it | `labInvalid` triggers a time-attack penalty at the finish; a clean finish from a mid-lap start must still be paid. |
| Progress counts from the start gate | `StartGate` / `WaypointsReached` / `WaypointTarget` | Gates skipped by a mid-lap start are not driven progress. Also fixes the diagnostic station-start overcount noted in `NORDSCHLEIFE-REVIEW.md`. |
| `spawnStation` in episode records | `EpisodeRecord.cs`, `EpisodeLog.cs` | Per-region crash analysis. |
| Circuit launch params | `rallylab/worker.py` `circuit_training_launch()` (~L435), `CIRCUIT_TRAINING_SECONDS = 180` | Training on circuits uses `spawnProfile: distributed`, `episodeSeconds: 180`; stored as `circuitTraining` in `run-manifest.json`; resume refuses if it changes. Eval/Watch keep fixed start + 1800 s. |
| Trainability gate | `Tools/circuit_trainability.py` | Sets `manifest.trainability.stage = "validated"` only if all 18 `Tools/circuit_probes.py` cases exist for this course, ran on the current build, have no missing expected surfaces / wrong-grip samples, non-barrier cases end `TimedOut`/`Finished`, and the barrier case recorded a contact. `api.py` `review_course` (L307) still refuses circuits without it — unchanged. |
| Zero-hazard audit skipped for circuits | `rallylab/api.py` L249 | Circuits report their 309 barriers as `obstacleObjects`; the procedural "no hazards" audit rejected them. |
| **Road/kerb tag fix** | `TrackGenerator.Circuit.cs` L262 | See 3.1. |
| `RALLY_OBS_DUMP` diagnostic | `RallyAgent.cs` `DumpObservations` (~L1100) | Env-gated: appends JSON lines with the 28-float vector observation, every ray's hit object/tag/fraction, and the action for the first decisions of each episode. Inert unless the env var is set. |
| Tests | `tests/test_circuit_trainability.py` | Launch params and gate evaluation. |

## 3. What went wrong on the way (resolved or understood)

### 3.1 Pavement tagged `TrackBoundary` — fixed
The circuit mesh builder tagged every non-barrier collider `TrackBoundary`, including
`Road_*` and `Kerb_*`. Procedural stages tag only terrain (`TrackGenerator.cs` ~L1210).
Ray sensors (`ForwardSensor` 60 m/25°, `WideSensor` 45 m/80°, detectable tags
`Obstacle,TrackBoundary`, zero vertical offset) therefore reported a boundary dead ahead
on every rise. Now only `Ground*` is `TrackBoundary`, barriers/overpass are `Obstacle`,
road/kerbs/markings untagged. Verified in `Assets/RallyLabGenerated/6e9c99ec43ab40f5-20261004/course.prefab`.
This required a player rebuild + circuit re-import (new course id) + re-running the 18 probes.

### 3.2 Fine-tuning from the Crests checkpoint failed — understood, abandoned
Parent `a2ec2da351d5…` (Crests bare-road overnight, 6M steps) finishes its home course
20/20 at ~95 mph but on the Nordschleife stalls at the start (fine-tune run
`985250c22f5343f2`: 96 % stalled after 630k steps, reward flat ≈ −20; stopped).
The obs dump (scratchpad, not in repo) showed vector observations were sane from tick 5,
but `WideSensor` sees barriers (tag `Obstacle`) ~9 m away on both sides. That parent's
course had zero obstacles, so the `Obstacle` channel was always 0 during its training;
with `normalize: true` a never-active input explodes once non-zero. Most likely cause,
not formally proven (the only obstacle-trained checkpoint, Gentle/Rocks `e3e404af0512`,
is itself weak on the current build). Conclusion: train circuits from scratch.

### 3.3 First observation of every episode is stale — minor, open
At tick 0 `GetObservations()` reflects the car's pre-reset pose (`rb.position` is set in
`OnEpisodeBegin` but the transform isn't synced until the next physics step). On the
Nordschleife this made tick 0 read heading error 1.0 / next waypoint 90° left because the
prefab sits at the origin. Self-corrects at the next decision (tick 5). Affects all courses.
Low priority; a `Physics.SyncTransforms()` / setting `transform` alongside `rb` at reset would fix it.

## 4. OPEN BUG TO FIX: cars flip within 1 s of a distributed spawn

### Evidence (job `21cb2eeff33940d3`, first 4,419 episodes)
- **893 episodes (20 %) end within 25 m of `spawnStation`**: 839 `RolledOver`, 54 `HitObstacle`.
- The rollovers end at **exactly 1.0 s** — i.e. the first physics step after
  `respawnGraceSeconds = 1.0` (`RallyAgent.cs` L121; `InGracePeriod` L1495 suppresses both
  `CheckTerminalConditions` L1421 and collision handling L1337).
- Record fields at end: `wheels` 0–2, `flight` ≈ 0.5–0.87 (airborne most of that second),
  `peakSpeed` 8–18 mph from standstill, `off` ≈ 0 (on centreline),
  but forensic `upright` ≈ 0.70–0.78 while the terminal check requires `up·Y < 0.1`.
  The forensic snapshot is sampled at 20 Hz, so the car goes from ~45° to inverted in < 50 ms
  — a physics blow-up, not a driving rollover.
- **Spread over the whole lap**: every 250 m bucket of spawn stations shows it at
  ~25–30 % of starts (e.g. 750 m: 19/60, 4,250 m: 17/68, 11,250 m: 16/63). Döttinger Höhe
  (a straight, 18.0–18.7 km) has ~50 of them. Not tied to particular corners.
- **Did not occur in the first ~1,000 episodes** (policy then did nothing → all `Stalled`).
  It appeared once the policy started applying throttle/steer during the first second
  (end-of-episode actions vary in sign; no single input pattern). So it needs **both** a
  distributed circuit spawn **and** non-zero inputs during the settle.
- Not reward hacking: time-attack charges forfeited clock (`ChargeForfeitedTime`,
  0.12/s × ~179 s ≈ −21.5 plus `failurePenalty`), so dying early is strongly negative.
- The 18 physics probes (waypoint-follow from station starts, 6 m/s target) never flipped,
  but they run at `timeScale 1` with a smooth heuristic controller; training runs at
  `timeScale 10` with arbitrary policy inputs.

### Relevant spawn path (`RallyAgent.OnEpisodeBegin`, L645–750)
1. `vehicle.ResetVehicle()` (zeroes velocities, neutral gear, resets tyres/suspension) —
   happens **before** the pose is changed.
2. `StartAtCircuitStation`: `spawn = CircuitSurfacePoint(station, 0)`,
   `forward = CircuitTangent(station)`.
3. `RoadUnder(spawn)` (L753): `RaycastAll` from `spawn + 10 m` down 30 m, keeps the nearest
   hit whose collider name starts with `Road_`, `Kerb_` or `Ground`; sets `spawn = hit.point`,
   `up = hit.normal`.
4. `forward` projected onto that plane; `rb.position = spawn + up * spawnHeight` (0.05 m),
   `rb.rotation = LookRotation(forward, up)` (L743).
   Procedural stages use the same path but always spawn at a flat-ish stage start;
   circuit spawns land on slopes up to Steilstrecke grade, camber, crests and bowls.

### Hypotheses to test (in rough order)
1. **Suspension/tyre state not consistent with the new pose.** `ResetSuspension()` runs
   before the teleport and the car is dropped 5 cm onto a sloped/banked surface; with throttle
   or brake applied while wheels are unloaded, the tyre model (`PacejkaTireModel`) or
   suspension (`DynamicSuspension`, bump-stop damping 3000 after the July rewrite) may
   produce a large impulse on first contact.
2. **Wrong surface picked / spawn inside geometry.** The nearest `Ground*` hit could sit
   above the pavement where terrain and road meet (shared terrain + shoulder strips), or the
   ray could miss the road and land on terrain below it, putting the car inside a collider →
   depenetration launch. Compare `RoadUnder` hit name and `hit.point` vs
   `CircuitSurfacePoint` for affected stations.
3. **Pose applied to `rb` only.** Transform isn't synced until the next step (see 3.3);
   wheel raycasts / suspension in the first step may use the old pose. Same mechanism as the
   stale tick-0 observation.
4. Velocity zeroed before the teleport (step 1) rather than after — interpolation or a
   residual from the previous episode's last step.

### How to reproduce (no training needed, but requires no active job holding `compute.lock`)
- Use `Tools/circuit_probes.py` `run()` as the template: it launches the managed player with
  `RALLY_CIRCUIT_REVIEW_STATION=<station>`, which goes through the same
  `StartAtCircuitStation` + `RoadUnder` path. Use `mode='fixed-input'` with e.g.
  `controlSteer=±0.8`, `controlDrive=±0.6`, and set `timeScale: 10` in the launch to match
  training. Try stations from the evidence: 750–1,000, 4,250–4,500, 11,250–11,500,
  18,000–18,700. Also try `controlDrive=0` as a control (should not flip).
- `RALLY_CIRCUIT_PHYSICAL_PROBE=<file>` gives per-sample wheel contacts; the trajectory
  file gives pose every 5 ticks. Log the first ~100 physics steps densely if needed.
- Quick stats from training records:
  ```python
  import json,glob
  r=[json.loads(l) for p in glob.glob('.rally/jobs/21cb2eeff33940d3/episodes/*.jsonl') for l in open(p)]
  bad=[e for e in r if e['outcome']=='RolledOver' and abs(e['station']-e['spawnStation'])<25]
  ```

### Fix direction (pick after reproducing)
- Make the spawn settle physically valid on any slope: e.g. apply the pose to both
  `rb` and `transform` (+ `Physics.SyncTransforms()`), reset velocities/suspension **after**
  placement, and hold controls at zero (or handbrake) during `respawnGraceSeconds` so the
  car settles before the policy drives. Ignoring actions during grace is the cheapest
  robust fix but changes what the agent experiences in the first second on **all** courses
  — guard it to imported circuits or confirm procedural behaviour is unchanged.
- Do **not** hide it by widening the grace period or excluding `RolledOver` near spawn.

### Acceptance
- Fixed-input probes at the stations above, with throttle/steer applied from tick 0 at
  `timeScale 10`, stay on four wheels.
- A short training run (≥ 300k steps, from scratch, distributed starts) shows spawn-site
  failures (`|station − spawnStation| < 25` and outcome ≠ Stalled) at < 1 % once the policy
  starts moving (the bug appeared after ~1,000 episodes, so run long enough to pass that).
- `./check-compile.sh` clean; Python tests pass; any change to the player requires a
  rebuild (`POST /api/build`), and if the course bundle content changes, re-import
  (`POST /api/circuits/import`), re-run `Tools/circuit_probes.py --course <id> --prefix <p>`,
  `Tools/circuit_trainability.py --course <id> --prefix <p>`, then `POST /api/courses/<id>/review`.
  A spawn-only code change should not need a re-import (the course bundle is unchanged),
  but `circuit_trainability` checks probes against the *current build hash*, so re-run probes
  and the gate anyway before a new training run.
- Then restart training from scratch (same spec as `21cb2eeff33940d3`, new seed is fine) —
  only with the user's go-ahead.

## 5. Other known issues (not blocking)

- **Barrier hit forensics distance is wrong**: barrier contact records `hitDist`/`hitLat`
  ≈ 1324 m (`stage-c-probe-barrier`). Forensics only; rewards unaffected.
- **`Tools/NORDSCHLEIFE-REVIEW.md` is stale**: describes Stage A/B; Stage C (full-route
  detailing, all 16 regions passed) finished later — see
  `.rally/circuit-review/stage-c-final-inspection-findings.json`. Physical review there is
  empty; the 18-probe suite in section 2 is the physical validation actually performed.
- `.rally/courses` holds ~15 superseded Nordschleife builds at ~75 MB each (gitignored).
- Deferred scope (not needed to train): 16-sector course library, full runtime fixture
  matrix (reverse / missed gates / teleport / seam), all-camera 1080p review, reload cycles,
  measured-width detailing.
- Crash hot spots in the current run (last 1,500 episodes), for later reward/curriculum work:
  barrier hits at Hohenrain (final chicane), Steilstrecke, Aremberg, Kallenhard; rollover
  counts are contaminated by the spawn bug until it is fixed.

## 6. Conventions and gotchas

- `.rally/` is gitignored; course/job/checkpoint state lives there.
- Training throughput ≈ 430 steps/s on 2 workers at `timeScale 10`.
- Unity's `Editor.log` can hold stale errors mid-refactor; `./check-compile.sh` is the authority.
- Evaluation API: `POST /api/evaluations {"checkpoint","courses":[...],"attempts","deterministic"}`;
  training: `POST /api/runs`; stop: `POST /api/jobs/<id>/stop`.
- Launching the player yourself (as done for the obs dump): copy a job's `launch.json`,
  set `courseBundle/courseId/output/attempts`, run the managed executable from
  `core.build_status()['executable']` with `RALLY_LAB_LAUNCH=<launch.json>` and
  `-batchmode -nographics -logFile <log>`.
