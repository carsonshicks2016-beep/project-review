# Training the rally driver

## One-time setup  (already done — this is the record of how)

The Python `mlagents` package has to match the Unity package (`com.unity.ml-agents`
4.0.3 in `Packages/manifest.json`), which means `mlagents==1.1.0`.

A plain `pip install mlagents` does **not** work on Apple Silicon. The recipe below is
what actually resolves; the reasoning is in "Why the pins" so nobody undoes it.

```bash
/opt/homebrew/bin/python3.10 -m venv .venv

.venv/bin/python -m pip install --ignore-requires-python \
  "numpy==1.23.5" "protobuf==3.20.3" "grpcio==1.47.5" "tensorboard==2.11.2" \
  "torch==2.2.2" "onnx==1.15.0" "h5py" "Pillow" "pyyaml" "six" "attrs" \
  "huggingface-hub" "cattrs>=1.1.0,<1.7" "cloudpickle" "filelock"

.venv/bin/python -m pip install --ignore-requires-python --no-deps \
  "mlagents-envs==1.1.0" "mlagents==1.1.0"

.venv/bin/python -m pip install "setuptools<81" "gym==0.26.2" "pettingzoo==1.15.0"
```

Verify:

```bash
.venv/bin/mlagents-learn --help
```

### Why the pins

Four separate things bite, in order:

1. **Python 3.10, not 3.14.** `python3` on this machine is 3.14, and mlagents 1.1.0
   declares `Requires-Python >=3.10.1,<=3.10.12`. Homebrew's python@3.10 is 3.10.20 —
   above that upper bound, hence `--ignore-requires-python`. The bound is patch-level
   conservatism; 3.10.20 shares the `cp310` ABI so every wheel is identical.

2. **grpcio has no arm64 wheel at the pinned version.** `mlagents-envs` wants
   `grpcio<=1.48.2`; `tensorboard>=2.14` wants `grpcio>=1.48.2`. That forces *exactly*
   1.48.2 — and the macOS arm64 wheels skip straight from 1.47.5 to 1.51.3, so pip
   falls back to a source build that fails. Breaking the deadlock needs grpcio 1.47.5
   (which has a wheel and satisfies mlagents) plus tensorboard 2.11.2 (the newest that
   only asks for `grpcio>=1.24.3`), installed with `--no-deps` on the mlagents packages
   so the resolver cannot re-introduce the conflict.

3. **`setuptools<81`.** `mlagents/torch_utils/torch.py` does `import pkg_resources`,
   which setuptools 81 removed. Anything newer fails at import with
   `ModuleNotFoundError: No module named 'pkg_resources'`.

4. **torch 2.2.2**, per the package's own `Installation.md` (`torch~=2.2.1`). Left
   alone, pip resolves a far newer torch that mlagents 1.1.0 was never tested against.

`gym` and `pettingzoo` are only needed for the optional gym/pettingzoo wrappers, not
for `mlagents-learn` itself.

## Build the scene

In Unity: **Rally → Training → Build Training Scene**.

This creates and saves `Assets/Scenes/RallyTraining.unity` with the track, the car, the
agent, the sensor and the camera all wired up, and adds it to the build settings. It is
reproducible — re-running it rebuilds the scene from code, so the setup lives in
`Assets/Editor/TrainingSceneBuilder.cs` rather than in a scene file nobody can diff.

## Train

For a real run, use the script — it is `mlagents-learn` plus the three things that have
each ended a run before:

```bash
./train.sh rally11
```

It refuses to start against a player older than the scripts (which is how a policy once
scored 85 % against a car that had since been rewritten), refuses to start if
`keep_checkpoints` is too small for the step budget (which is how rally08 deleted its own
best policy), and wraps the trainer in `caffeinate` (which is what stops the Mac sleeping
mid-run and killing training with no error anywhere).

To drive it by hand instead — note the `.venv/bin/` prefix, the venv is not on `PATH`:

```bash
.venv/bin/mlagents-learn config/rally_ppo.yaml --run-id=rally01
```

Wait for `[INFO] Listening on port 5004. Start training by pressing the Play button`,
then press Play in Unity. Stop with Ctrl-C; resume the same run with `--resume`, or
start over with a new `--run-id` (reusing one without `--resume` needs `--force`).

## Measure

Training tells you the *rate* of each failure. It cannot tell you why, because the stage,
the difficulty and the policy are all moving at once. Evaluation holds all three still:

```bash
./evaluate.sh --episodes 120 --rocks 2       # 120 fresh stages, fixed difficulty
./evaluate.sh --episodes 120 --rocks 0       # the control: no rocks at all
./evaluate.sh --seeds 20260727,4711          # replay two specific stages
```

Each episode records where and how it ended — station along the stage, speed, lateral
offset, road curvature, airborne time, what was struck, the widest clear gap that existed
there, and how long the obstacle was in the sensor's view. Read it back with:

```bash
.venv/bin/python analyse-episodes.py results/eval/<timestamp>
```

The seed of every episode is in the log, so anything interesting can be replayed exactly.

## Watch an old run

Policies are only loadable against the observation vector they were trained with, so each
time the agent gains a sense the previous era becomes unwatchable — rally08 already is.
Ghosts get around that by recording where the car went rather than the brain that drove
it. The best lap of every run is kept in `results/ghosts/`, and
**Rally → Watch → Race Ghosts on a Stage** puts them on the same stage together.

Watch it two ways — they answer different questions:

```bash
tensorboard --logdir results
```

TensorBoard owns the reward and loss curves. **Rally → Training → Training Monitor**
owns the thing TensorBoard cannot show you: *why* episodes are ending. A flat reward
curve looks identical whether the car is rolling over on turn-in, timing out halfway
down the stage, or driving into the scenery — and those need three different fixes.
Dock it next to the Game view and leave it open.

## Use the trained policy

Training writes `results/rally01/RallyDriver.onnx`. Drag it into the car's
**Behavior Parameters → Model** field and press Play — it will drive itself with no
trainer attached.

## Notes

- `BehaviorName` is `RallyDriver` on both sides. Change one, change the other.
- The vector observation is 28 floats, defined by `RallyAgent.VectorObservationSize`.
  If you add an observation, update that constant and rebuild the scene — Behaviour
  Parameters reads it from there.
- **The stage changes every 15 episodes** (`RallyAgent.stageRefreshEpisodes`), and that
  is not optional decoration. `TrackGenerator.GenerateTrack` runs once, from its own
  `Awake`, so `TrackGenerator.randomiseSeed` on its own draws one random stage per
  *process* — with `--num-envs 6`, six stages for a ten-million-step run, which is as
  memorisable as one. The agent therefore sets a fresh seed itself and rebuilds. The
  scene asset keeps its fixed seed so it stays reproducible and diffable.

  This is what a policy has to beat now, and it is much harder than what rally05 faced.
  Do not compare new runs against rally05's 107 mean reward: that was measured on a
  single memorised course, by a policy that could not perceive the road at all, and it
  reached 0 waypoints the moment anything moved.
- Rocks on the road ramp in through the `obstacle_density` curriculum in
  `rally_ppo.yaml`, read on each stage refresh. Note this is only about the racing
  line — the ~259 trees and ~77 boulders of scenery are tagged `Obstacle`, end the
  episode, and are present at every lesson. There is no "clean stage" setting.
  The lessons are gated on `progress` (fraction of `max_steps`) rather than `reward`,
  because there is no trustworthy reward scale for random stages yet. Once one run has
  established it, switch to `measure: reward` so difficulty tracks competence instead
  of elapsed steps.
- **Two world-generation bugs invalidated every failure statistic before 30 July.** Both
  were found by measuring rather than reasoning, and both are fixed:

  1. *Stale scenery and terrain.* Regenerating a stage replaced its trees, terrain and
     backdrop with `Destroy()`, which in play mode is deferred to the end of the frame
     while the replacement is built immediately — and `Transform.Find` then returned the
     corpse rather than the live object. Three scenery roots accumulated, so every stage
     carried two previous stages' trees and landforms, standing wherever those roads used
     to go. Stages whose own report said the nearest tree was 14.4 m away were ending
     episodes against trees 1.7 m from the centreline.
  2. *The agent measured against the wrong centreline.* Its road observations projected
     onto the polyline through the waypoints, while the road mesh, its edges and every
     prop are built on the Catmull-Rom spline through those same waypoints. Measured over
     25 stages the two disagree by 2.7–4.4 m, on a road with a 6 m half-width.

  What they cost, measured on the unchanged rally09 policy over 150 random stages:

  | | before | after |
  |---|---|---|
  | finished, 2 rocks/100 m | 14 % | **32 %** |
  | finished, no rocks | 27 % | **89 %** |
  | rolled over | 8.7 % | **1.3 %** |
  | hard impacts | 18 % | **2 %** |
  | tree and boulder strikes | 35 | **0** |

  The rollover rate that "sat at 15–20 % across every run and resisted everything" was
  the car hitting scenery and landforms that no longer existed. Do not compare any
  measurement taken before this against one taken after.
- **The physics runs at 100 Hz, not 50.** `ProjectSettings/TimeManager.asset` has a
  0.01 s fixed timestep, and every comment in the codebase claimed 0.02 for most of the
  project's life. `Agent.MaxStep` counts physics steps, so the episode described as
  "6000 = 120 s" was really 60 s — half the intended clock, on a 1 km stage, which
  penalised exactly the careful driver the reward is trying to produce. `MaxStep` is now
  12000 and `RallyAgent` prints the real rate at startup so it cannot drift again.
  Decisions are every 5 physics steps, so 20 Hz rather than the 10 Hz long assumed.
- Set `VehicleController.allowKeyboardFallback = false` before a long run so a stray
  keypress in the editor cannot fight the policy.
- Throughput: one car per scene is slow. The cheap fix is a standalone build plus
  `--num-envs 8` on `mlagents-learn` — no scene changes needed.
