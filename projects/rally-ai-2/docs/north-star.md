# RallyAI — North Star

**Status:** the authoritative statement of what this project is. Plans, tickets, and
session notes serve this document; where they disagree, this wins.
**Audience:** Carson, and any agent picking the project up cold.
**Supersedes:** the scope sections of `physics-driving-visuals-plan.md` and
`gallery-dashboard-plan.md` (their implementation detail still stands).

---

## 0. The pitch

**A reinforcement-learning agent teaches itself to drive a rally car flat-out.**

You give a PPO policy the wheel of a Group-A-era Mitsubishi Evo — red-and-white
tobacco-livery homage — and drop it on a procedurally generated rally stage it has
never seen. Gravel, snow, crests, hairpins, trees a metre off the racing line. It
gets no racing line, no waypoints, no demonstrations. It gets what a driver gets:
what it can see down the road, what it feels through the car, and a stopwatch.

Then you watch it learn, rendered as a PlayStation-1 rally game.

The product is the *watching*. A training curve is a chart; a car that spends ten
thousand episodes understeering into trees and then, one evening, starts trail-braking
into a hairpin and catching the slide on opposite lock — that is the thing worth
building. Every decision below serves legibility of that arc.

## 1. What "done" looks like

A single command brings up the Gallery. Two modes:

**WATCH.** A cleanly rendered early-3D rally stage, the Evo mid-slide, faceted dust
hanging in the air and chunky polygonal trees framing the road. Chase, bonnet, and
sideline cameras. A stopwatch, a pace-note call, a gear indicator. It plays a replay
file, or streams a live agent, or streams *you* driving on the keyboard. Nothing on
screen says "machine learning."

**TRAIN.** The same stage in a viewport, but now it is the agent driving, live, while
the run happens. Reward sparkline, stage-completion rate, current curriculum tier, the
policy's own sensor rays drawn over the road so you can see what it sees. Arm, start,
stop. Runs survive a browser refresh because the server owns the job, not the tab.

The finished claim, stated plainly:

> On a stage generated from a seed it has never encountered, at the hardest curriculum
> tier, the trained policy finishes clean — no crash, no off-course — on the large
> majority of attempts, at a pace a competent human on a keyboard cannot match.

Plus: seed `N` always produces the same stage; a replay always plays back identically;
a checkpoint's numbers can be reproduced from the checkpoint.

## 2. Principles

These are invariants. Violating one is a bug even if the code works.

1. **Python owns the truth.** Physics, sensing, reward, and termination live in
   `packages/sim`. The browser renders state it is handed and sends inputs. It never
   steps the world. This is what makes replays, training, and live view the same thing.
2. **One tunable surface per concern.** If a number lives in two files, one of them is
   a lie waiting to happen. (This is not hypothetical here — see §12.)
3. **The agent may only use what a driver could have.** No global stage array, no
   privileged distance-to-finish oracle beyond what pace notes give a co-driver, no
   peeking at the reward function's internals. If a human with these senses could not
   drive the stage, the senses are wrong.
4. **Anything that can kill the agent must be visible to the agent.** A termination the
   policy cannot see coming is not a lesson, it is noise in the value function.
5. **Spectacle is a feature, not a garnish.** The stylized early-3D look, the audio,
   the pace notes, the camera work — these are the deliverable, equal in rank to the
   training loop.
6. **Every claim is measured.** "Faster," "better," "more stable" ship with the number
   and the method. Benchmark before and after, in the same session.
7. **Branding is a viewer skin, not simulator truth.** For this personal project,
   WATCH may use the historic Subaru/555-era marks the owner explicitly approved.
   Branding never changes the `evo_rally` physics preset or replay contracts.

## 3. Architecture

```
packages/sim/          Python. The world and the brain.
  physics/             Vendored Supra Ai 2 dynamics + the RallyAI coordinate bridge
  track/               Stage geometry: corridor, elevation, camber, surfaces, obstacles
  stage/               Procedural generator (seed -> stage)
  sense/               Observation assembly (the agent's senses)
  env/                 Gymnasium env: obs, action, reward, termination
  train/               PPO, vectorisation, curriculum, checkpoint management
  control/             FastAPI control room: job control, live drive, metrics stream
packages/viewer/       TypeScript. Vite + Three.js. Renders; never simulates.
packages/shared/       JSON schemas + authored stages. One copy, both sides read it.
```

**The three contracts.** Everything crossing a boundary is one of:

- **Stage JSON** — centerline (x, y, z), per-point width and camber, surface segments
  with µ, pace notes, obstacles. Seed-reproducible. Schema in `shared/schemas`.
- **Replay JSON** — stage reference plus a frame array: pose, per-wheel state, slip,
  drivetrain, surface, FX triggers, pace call. Versioned. A replay is the *only*
  artifact needed to reproduce a run visually.
- **Metrics JSONL** — one line per training event: timesteps, reward, completion rate,
  curriculum tier, checkpoint path. Durable on disk, tailed to the browser over a
  WebSocket. The file is the truth; the socket is a convenience.

Any of the three should be readable by a person in a text editor and diffable in git.

## 4. The car

**Group-A / WRC-era Mitsubishi Evo.** Turbocharged inline-four, AWD with a centre
differential, ~1250–1300 kg, snappier and more nervous than the Impreza preset
currently in the tree — it should reward commitment and punish laziness.

It lives as a `CarSpec` preset (`evo_rally()`) beside the existing `impreza_rally()`.
The spec is the single tuning surface for handling: mass, weight distribution, yaw
inertia, Pacejka coefficients, relaxation length, differential coefficients, torque
curve, gearing, boost response, aero. Nothing about how the car drives lives anywhere
else.

**Livery.** Deep works blue, yellow 555-era graphics, a white door number, Subaru
rear/wing marks, gold wheels, and gravel grime low on the bumper and quarters. This is
an explicitly approved personal-viewer skin; the simulated car remains `evo_rally`.
The read at 30 metres and at speed is what matters, so large colour blocks and marks
take priority over tiny sponsor stickers.

**Feel targets** (human-noticeable, not telemetry-perfect):

| Surface | What the driver should feel |
|---|---|
| Gravel (µ ≈ 0.65–0.78) | Mild understeer on greedy throttle; recoverable slides; visible scrub; dust plumes |
| Tarmac (µ ≈ 0.90–1.05) | Sharp turn-in, planted, brake later and harder |
| Snow (µ ≈ 0.28–0.45) | Float. Slow yaw build, long recovery, understeer that becomes oversteer if you dump the throttle |
| Lift-off mid-corner | Front loads, rear lightens, snap rotation — catchable with prompt opposite lock |
| Crest at speed | Light, then airborne, then a landing that must be straight or it costs you |

## 5. The world

**Stages are generated, not authored.** Hand-authored stages exist as fixtures and
demos; the agent's job is stages it has never seen. Generation from a seed produces:

- a **centerline** with real elevation — climbs, descents, and crests sharp enough to
  actually launch a car at speed, not decorative sine hills;
- **variable width and camber**, banking into turns, off-camber where it should hurt;
- **surface segments** — gravel, tarmac, snow, mud, with transitions the agent must
  read ahead for;
- **corner geometry with intent** — a hairpin, an ess sequence, a long fifth-gear sweep,
  a crest-into-braking-zone. Composed from a vocabulary of corner archetypes with
  plausible sequencing, not a random walk that happens to bend;
- **pace notes** derived from the geometry, exactly as a co-driver would call them;
- **obstacles** — trees, rocks, banks — placed with a hard rule: *nothing that can
  terminate the agent sits inside the drivable corridor unless the agent can sense it.*

Difficulty is one scalar the curriculum turns: length, width, µ, corner severity,
elevation aggression, and obstacle proximity all move together.

## 6. Physics

The vendored Supra Ai 2 model is the dynamics, and it is already more capable than the
current bridge exposes. It carries per-wheel contact patches with slip angle and slip
ratio, one combined-slip Pacejka budget shared correctly between lateral and
longitudinal demand, tyre-force relaxation length so grip builds transiently, dynamic
load transfer with roll distribution and load-sensitive peak µ, a sub-stepped
engine → clutch → gearbox → LSD → wheel driveline with turbo spool, drag against the
full velocity vector so big slip angles scrub speed, and a 2.5D road plane where slope
gravity is a real body force.

**It also has a complete airborne model** — crest-triggered takeoff, ballistic flight,
friction gated off in the air, landing. Rally without jumps is not rally. Wire it up:
takeoff and landing become first-class events the agent senses, is rewarded and
punished for, and the viewer dramatises.

Non-negotiables: fixed timestep, deterministic given seed and action sequence, physics
identical between training, evaluation, live drive, and replay. One code path.

## 7. The agent

### Senses

Modelled on the Supra Ai 2 `SensorSuite`, adapted to a point-to-point stage. Four
groups, all normalised to roughly [-1, 1], all reproducible for on-screen debug draw:

1. **Vision** — raycast beams fanned ahead, returning distance to whatever stops the
   car: corridor edge, tree, rock, bank. This is the group the current build lacks
   entirely, and it is the one that makes "without crashing" learnable.
2. **Proprioception** — speed, body-frame velocity, lateral and longitudinal g, yaw
   rate, per-wheel slip and load, grip budget in use, gear, RPM, boost, steering angle,
   position across the corridor.
3. **Look-ahead** — signed centerline curvature at several distances scaled to current
   speed, plus surface µ ahead, plus the current pace note. The faster it goes, the
   further ahead it must read.
4. **Terrain and air** — road slope under the nose, camber, vertical speed, height above
   the road, airborne flag, and grade/crest preview at the same look-ahead distances.
   This is the jump detector: with its own speed, the policy can predict takeoff, set up
   the landing, and brake before a downhill corner.

Rule from §2.4, restated because it is the one most easily broken: if you add a way to
die, you add the sense that sees it coming, in the same commit.

### Controls

`[steer, throttle, brake, handbrake]`, continuous. The handbrake is not optional — the
goal says *by any means*, and a rally car pivoting a hairpin on the handbrake is the
single most recognisable thing in the sport. Steering is rate-limited at the physics
layer, not clamped in the policy.

### Reward

The objective is **finish the stage, clean, in minimum time**, and the shaping exists
only because that signal alone is too sparse to bootstrap. Carry over what Supra Ai 2
learned the hard way:

- **Progress** along the centerline, the dense backbone.
- **An edge taper.** Every positive term scales 1 → 0 across the outermost metre of the
  corridor and is zero off it. Without this the policy learns to ride the verge
  flat-out, farming centerline progress while flicking back inside just often enough to
  dodge the off-course timer. This exploit is not theoretical — it is why `edge_band`
  exists in the Supra reward, and the current RallyAI reward is wide open to it.
- **Super-linear speed on straights.** A reward linear in speed makes a safe cruise
  nearly as profitable as commitment, and the policy learns to lift early. Reward
  `(v/v_ref)^2`, gated by how straight the road is at a *speed-scaled* look-ahead, plus
  a flat bonus for holding full throttle there. The gate closing earlier at high speed
  is what teaches it to lift in time for the corner at the end of a long straight.
- **Time.** Stage time must be a real fraction of the return, not a rounding error.
  Today's finish bonus makes pace roughly 0.3% of a stage's total return — the policy is
  being asked to hurry by a whisper. Fix the magnitude, and check it after every reward
  edit by integrating the terms over a full episode.
- **Landings.** Reward a straight, settled landing; punish landing sideways or nose-first.
- **Smoothness**, lightly. Enough to stop input chatter, not so much that it can't
  throw the car at a hairpin.
- **Terminal:** large penalty for crashing or leaving the course, large bonus for
  finishing, scaled by time.

### Terminations

Crash, off-course beyond the grace timer, finish, and time limit — plus **stuck or
spun**, which the current build lacks. A car sitting on the road pointed backwards
currently burns up to 150 seconds of simulated time per episode collecting nothing.
That is rollout budget spent on a dead episode, and it is exactly how the shipped demo
replay dies. Detect no-progress and sustained large heading error, and end it.

## 8. Training

Port the Supra Ai 2 learning stack rather than reinventing it. Stock library defaults
will not get there, and the lessons are already paid for.

**The trainer.** PPO in PyTorch: GAE, clipped surrogate, entropy bonus, and — the parts
that matter and that defaults get wrong — correct bootstrapping of time-limit truncation
as distinct from real termination, a KL trust-region guard so one bad update cannot
destroy a competent policy, value-loss clipping sized to the return scale, and
learning-rate and entropy annealing for long runs.

**Throughput is a first-class feature.** Parallel environments across worker processes,
with workers that get revived rather than taking the run down. Observation
normalisation with running statistics that are checkpointed alongside the weights — a
policy restored without its normaliser is a different policy. Track steps/second as a
tracked metric; a regression in it is a regression.

**Curriculum by mastery, not by clock.** Advance difficulty when rolling stage-completion
clears a threshold, never on a timestep schedule. A schedule marches a policy onto
stages it cannot drive and teaches it that the world is arbitrary.

**Staged pipeline.** Successive objectives on the same policy, as in Supra Ai 2's
foundation → flow → fast → finish → frontier: first survive and complete, then carry
speed, then commit, then optimise the last seconds. Each stage a different reward
weighting over the same env.

**Checkpoints.** Keep a hall of fame, not a single file: best overall, cleanest run,
fastest stage, furthest progress. Validate on load — finite weights, matching
normaliser, hash that matches the payload. A corrupted checkpoint that loads silently
costs a night of training.

## 9. The viewer

**Aesthetic: stylized early 3D, committed.** High-resolution low-poly rally diorama:
the nostalgia lives in the models rather than in a magnified framebuffer. Strong
flat-shaded planes, exaggerated silhouettes, chunky polygonal trees, broad painted
surface patches, aggressive draw distance and fog eating the horizon. The sky is a few
colours and angular shapes. A trace of distant vertex wobble is welcome; obvious pixel
grids, full-screen dither and crushed colour are not. Keep the car, road and HUD clean
and readable at speed.

**Audio, procedural:** engine note driven by RPM, load, and boost, with off-throttle
overrun and shift punctuation; surface-dependent tyre noise; gravel peppering the arches;
wind with speed; biome ambience per surface. Co-driver pace calls timed off the notes
the agent is reading.

**WATCH:** replay playback with scrub and speed control, live agent stream, live human
drive. Chase, bonnet, sideline. Stopwatch, split, gear, pace call, stage progress.

**TRAIN:** live viewport of the agent training, reward sparkline, completion rate,
curriculum tier, sensor debug overlay drawing the agent's rays and look-ahead points
over the road. Arm / start / stop against the server-owned job.

Both modes read the same replay and state formats. A live stream is a replay that has
not finished being written.

## 10. Evaluation

A run is scored by an evaluation harness, not by eyeballing the reward curve. Per
checkpoint, over a fixed set of held-out seeds:

- **completion rate** — finished, clean, no off-course
- **stage time** vs the generator's theoretical minimum for that seed
- **termination breakdown** — crash / off-course / spin / timeout, as counts
- **cleanliness** — time off-track, obstacle contacts
- **sector analysis** — which parts of which stages it is losing time in, and why
- **human baseline** — your own keyboard runs on the same seeds, as the bar

Held-out seeds are never trained on. Report generalisation, not memorisation. Store
each evaluation as JSON beside the checkpoint so any claim can be re-derived later.

## 11. Build order

Each milestone ends with something you can watch or measure. **[roadmap.md](roadmap.md)
expands this list into executable work items** — files, design decisions, required
measurements and traps per step. This section holds the intent; the roadmap holds the
plan, and is reconciled against the code rather than against this list.

0. **Foundation.** Repo in git. Throughput adequate for real experiments. *(Done:
   projection rewrite took the env from 285 to ~950 steps/s single-threaded.)*
1. **Parallel rollouts.** Worker-process vectorisation. Steps/second becomes a tracked
   number.
2. **Senses.** The full sensor suite, raycasts first, with the debug overlay in TRAIN so
   you can watch what it sees while it learns.
3. **Reward and terminations.** Edge taper, super-linear straight speed, real time
   pressure, stuck/spun detection. Verify by integrating terms over an episode.
4. **Trainer.** Port the PPO: truncation bootstrapping, KL guard, normaliser
   checkpointing, mastery-gated curriculum, hall of fame.
5. **First honest result.** A policy that finishes unseen generated stages at the easy
   tier, evaluated on held-out seeds. This is the project's first real milestone —
   everything before it is scaffolding.
6. **Stage generator v2.** Corner archetypes, real crests, deliberate difficulty scaling.
7. **Jumps and handbrake.** Airborne sensing and reward; fourth action; the physics is
   already there.
8. **The car.** `evo_rally()` spec and the approved blue/gold personal viewer skin.
9. **Spectacle pass.** Cameras, FX, audio, pace notes, the TRAIN overlay.
10. **The long run.** Staged pipeline to convergence, evaluated against the human
    baseline.

The temptation will be to do 8 first because it is the fun one. The car is a texture
swap and a spec file; it will still be a texture swap and a spec file after the agent
can drive.

## 12. Traps

Concrete, all observed in this codebase or its predecessor.

- **Dead tuning surfaces.** `physics/constants.py` defines 72 constants of which 9 are
  live after the Supra cutover, while its own header and the README both direct you to
  tune there. Delete it or gut it to what is real. A tuning knob wired to nothing is
  worse than no knob.
- **Duplicated data.** Stage JSON exists byte-identically in `shared/stages` and
  `viewer/public/stages`, synced by a `shutil.copy` buried in a demo script. Edit one,
  forget the other, and the renderer draws geometry the physics never stepped — which
  presents as a mysterious floating car. One copy.
- **Invisible killers.** The shipped `snow_rally_01` has 21 trees *inside* the drivable
  corridor, each an instant `-25` and episode end, against an 18-dim observation with no
  obstacle sensing at all. Fix by sensing, not by moving the trees.
- **Reward the agent can farm.** Assume every reward term will be exploited and go
  looking before training does. Verge-riding is the known one; there will be others.
- **Hand-tuned heuristic cascades.** The demo autopilot is 90 lines of stacked magic
  numbers whose relaunch gate requires a heading error under 50° — which a spun car can
  never satisfy, so it parks forever at 55% of the stage. Heuristics are for
  bootstrapping demos, never for anything load-bearing.
- **Documentation drift.** The root README once said the viewer was not started after
  WATCH had already shipped, while older livery rules contradicted the owner-approved
  personal skin. Update viewer status and presentation rules in the same change as the
  implementation so a later cleanup does not "fix" the code back to a stale goal.
- **Unmeasured optimisation.** Benchmark first. The projection rewrite looked like a
  clean 4-metre-accurate win until measurement showed it had silently swapped the
  reference curve out from under `lateral_error` and the road mesh.

## 13. Definition of done

- [ ] Seed → stage is deterministic and reproducible across machines.
- [ ] The agent's senses contain everything that can end its episode.
- [ ] Handbrake and jumps are in the action space and the physics, and used.
- [ ] Training runs multi-process, with throughput tracked as a metric.
- [ ] Curriculum advances on demonstrated mastery.
- [ ] Checkpoints carry their normaliser and validate on load.
- [ ] Evaluation on held-out seeds, stored as JSON, with a human baseline.
- [ ] The policy finishes unseen hard-tier stages clean, most attempts, faster than a
      competent human on a keyboard.
- [ ] WATCH looks like a PlayStation 1 rally game to someone who does not know what PPO
      is.
- [ ] TRAIN shows the agent driving live with its own senses drawn over the road.
- [ ] `README.md` describes what the code actually does.
- [ ] Approved branding remains presentation-only and never leaks into simulation truth.

---

*Related: [[Context/Architecture]] · [[Goals]] · [[Areas/Sim Physics]] ·
[[Areas/PPO Training]] · [[Areas/Viewer Gallery]] · `physics-driving-visuals-plan.md` ·
`gallery-dashboard-plan.md`*
