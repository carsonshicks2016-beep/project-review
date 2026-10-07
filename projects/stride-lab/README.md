# Stride Lab

A working, browser-based first experiment for the generalist-athlete idea: a shared controller for a 100 m sprint and a custom 60 m, five-hurdle course.

## Run

Requires Python 3 only to serve the files:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8765. The physics, training worker, visualization, and exports run on your device. No build or package installation is needed. Google Fonts is the only optional external asset request. A system-font fallback works offline.

Use **Run baseline**, select the saved shared example, or **Train a policy**. Train shared or event-specialist controllers from the same baseline. Inspect slow-motion replays, scrub the timeline, switch force/energy/height plots, change physical conditions, and run the sensitivity check. Export includes controller parameters, provenance, scores, full replay telemetry, and the latest robustness results. Data stays in the current page session unless exported.

## Train without the interface

Requires modern Node.js (22+ recommended). No dependencies:

```sh
node scripts/train.mjs shared 60 42 outputs/training-result.json
node scripts/train.mjs sprint 60 42 outputs/sprint-result.json
node scripts/train.mjs hurdles 60 42 outputs/hurdles-result.json
node --test tests/physics.test.mjs
```

The arguments are scope, generation count, random seed, and output path. A shared 60-generation run takes about 20 s. The browser and command-line runner use exactly the same simulation and search module. The test suite takes about 30 s, most of it re-deriving the bundled example from its seed.

## What is implemented

- A planar actuated spring-loaded inverted pendulum (SLIP), initialized at rest.
- A point body mass plus a swing limb carrying 12 % of body mass at 40 % of leg length, compression-only spring contact, damping, drag, and bounded axial drive.
- The swing limb is moved by a hip actuator bounded at 700 N on the limb's centre of mass, about 294 Nm of hip torque. It is never teleported: the foot is planted wherever the limb has actually reached, which lags the commanded placement by 6–8 cm. Results are flat for hip limits from 500 N to 1200 N, so the gait does not balance on that constant.
- Leg length is a state. It retracts and extends in flight at up to 6 m/s, and is frozen through stance, where the spring is loaded. Retraction therefore costs reach and delays the next contact rather than being free clearance. When the body is inside the reach circle at contact, the limb flexes to fit — the leg shortens to the hip-to-foot distance — instead of the foot being flung out to the reach distance.
- Leaving the friction cone slides the foot and dissipates energy instead of ending the trial. The condition is pure geometry for a massless point foot: |x − foot| ≤ µ·y. At µ = 0.12 the baseline still finishes, having slid 9.6 cm and lost 21 J.
- A 2 ms symplectic-Euler timestep, with touchdown solved to the exact crossing time inside the step. A stance therefore never opens with the leg already compressed, and contact cannot inject spring energy that the body did not have. `maxContactCompression` reports the largest violation in a trial; it is zero to machine precision.
- A closed energy ledger on every trial: `actuatorWork` in, `dissipated` (leg damper and drag) and `clampLoss` (force-cap truncation) out, and `energyResidual` for what is left. The residual is this integrator's discretisation error alone — under 0.03 % of actuator work at the default timestep, and it converges to zero as the timestep shrinks. A test asserts both the bound and the convergence.
- When the commanded foot offset is unreachable with a straight leg at the current hip height, the placement is clamped to the reachable distance instead of pre-loading the spring. These contacts are counted in `lowApexContacts`: 43 of 119 for the baseline, 10 of 58 for the saved example.
- Eight controller parameters optimized by seeded cross-entropy policy search (CEM).
- 32 population members; the incumbent is re-simulated under each generation's conditions, so elitism compares like with like and a lucky draw cannot be banked. Six elite controllers fit the next sample distribution.
- Training conditions are resampled every generation — body mass ±10 %, friction ±15 %, hurdle height ±0.05 m, and timestep from {2 ms, 1 ms, 0.5 ms}. Candidates within a generation share one set of conditions (common random numbers) so their scores are comparable. A controller that only works at one discretisation cannot score well.
- The returned policy is chosen on 8 fixed held-out conditions drawn from a separate stream, never on the generation's own draw. That held-out score is what the learning curve plots, and it is monotone by construction.
- Equal weight across events and across conditions for a shared policy. Each event specialist uses half as many physical trials per generation. Trial counts remain visible so comparisons are not misrepresented as budget-matched.
- Finish time measured only on successful trials. Failed trials show termination cause and distance.
- Actual computed force, velocity, CoM height, kinetic energy, gravitational energy, spring energy, and positive actuator work. Peak force is the magnitude of the ground reaction, not its vertical component alone.
- Deterministic sensitivity checks at half, quarter and eighth timestep, and for ground friction, body mass, and hurdle height.
- Accessible semantic controls, responsive canvas rendering, and optional feature-detected WebMCP read/replay tools.

## What is not implemented

This is a reduced mechanics prototype, not the full 3D, muscle-driven, 23-DOF athlete in the original concept. It does not simulate anatomical joints, muscle force–velocity behavior, limb inertia, 3D balance, fatigue, or pole vaulting. The upper body and joint poses are illustrative. Hurdle contact uses a lower clearance envelope controlled by retraction, not full limb collision. The gait structure and obstacle reflex are hand-designed; CEM optimizes their parameters. This is not discovery of human technique from scratch.

The custom hurdle course has hurdles at 12, 21, 30, 39, and 48 m. Default height is 0.35 m; this is not an official competition layout, and it was lowered from 0.60 m when retraction became a real leg-length change (see below). A fixed 30 s episode limit applies. The finish is determined by the point body's horizontal coordinate.

The contact-force cap is 5.5 body weights. Positive axial drive is limited to 2.4 kW normally and 4.2 kW during a hurdle takeoff. These are illustrative experiment settings, not validated human limits. Energy removed by the force cap is measured separately as `clampLoss` rather than vanishing. The mechanical-energy chart remains a mechanical quantity: it is not metabolic expenditure or a physiological efficiency estimate.

One planar leg length is shared by both limbs, so the model cannot represent hurdling technique — a lead leg extended forward and high while the trail leg tucks sideways. Clearance here means getting the whole body over the barrier on a retracted leg. The limb's mass rides at the body's height and translates horizontally; it does not swing on an arc, carry rotational inertia, or exert a moment on the trunk. Touchdown impact loss is not modelled: the limb coordinate is handed over between legs at a constant swing rate, which is exactly energy-neutral, rather than being mismodelled. Hip work is mechanical work, not a metabolic cost, and the hip is treated as regenerative — it recovers the energy it spends decelerating the limb, which real muscle does not.

## The articulated body

The spring-leg model above is a point mass with a leg attached. This is a body: nine links
and eleven degrees of freedom — pelvis position and pitch, two hips, two knees, two ankles,
two shoulders — with segment masses, lengths and radii of gyration taken from Winter's
anthropometric tables, so 75 kg is distributed the way a person's is rather than lumped.

Nothing holds it up. There is no spring standing in for a leg, no prescribed stance, no
gait phase. With zero torques it collapses in 0.27 s.

### Dynamics

`M(q) q̈ = τ + Σ JᵀF − bias(q, q̇)`, solved every step. The mass matrix is assembled from
segment Jacobians; the bias is the centripetal acceleration each segment carries when all
joint accelerations are zero, propagated down the tree. This is exact for the topology, not
an approximation of it.

Ground contact is a penalty spring and damper at ten points — pelvis, shoulders, knees,
hands and four sole points — with regularised Coulomb friction. Ten rather than two because
a body learning to balance spends most of training falling over, and a ragdoll whose torso
has no contact points simply sinks through the floor, which makes the failure carry no
information.

The spring force is applied whenever a point is below ground, so its stored energy is always
returned, and the damper is limited so a contact can never pull. There is no discarded-force
term for the ledger to hide in.

| Check | Result |
|---|---|
| Free flight, no contact | energy conserved to 0.005 %, drift exactly halves with dt |
| Dropped ragdoll | falls, lands, settles; resting pitch identical across an 8× dt range |
| Contact ledger through a full-body impact | 9.9 % → 4.1 % → 1.7 % → 0.39 % as dt quarters |
| Contact forces | never pull; friction never leaves the cone |
| Speed | 333k steps/s with the ledger off, 218k with it on |

Floor stiffness trades the ledger against penetration. The shipped 10 kN/m sinks 2.4 mm under
a standing body — about what a running track deflects — and closes the ledger to roughly 4 %
through an impact at the default 0.5 ms timestep.

### Control

The policy is one matrix: observation in, joint torques out. It sees torso pitch and rate,
all eight joint angles and rates, pelvis height and velocity, and which soles are loaded —
25 numbers — and returns eight torques, each clamped to a peak human joint torque (200 Nm at
hip and knee, 150 at the ankle, 60 at the shoulder). There is no phase machine, no reference
motion, and no hand-designed stride anywhere in it.

Training is Augmented Random Search: probe symmetric perturbations, keep the directions with
the strongest response, step along their reward-weighted average scaled by the spread of the
kept returns. Observations are normalised by what the agent has actually seen. Linear
policies are enough for locomotion under that normalisation (Mania, Guy and Recht, 2018), and
the search needs nothing but forward simulation, so a run is reproducible from its seed.

```sh
node scripts/train_balance.mjs 1200 42 outputs/balance.json
node scripts/train_walk.mjs 2000 42 outputs/walk.json
node --test tests/body.test.mjs tests/control.test.mjs
```

Open `dist/ragdoll.html` to watch it, compare against the zero-torque ragdoll, and shove it.

### Balance

Zero torque collapses in 0.27 s. The shipped policy stands.

The number that matters is not the training return. The first policy trained here scored 593
of a ~600 ceiling and stood its full episode, and it was not balancing — it had memorised one
torque sequence for one exact start. `scripts/probe.mjs` is what caught it: shoves from 20 to
160 N·s, bodies at 60 and 95 kg, friction at 0.5 and 0.25, soft and stiff floors, four
timesteps, and four poses never trained from.

| | memorised | shipped |
|---|---:|---:|
| clean start | 1/1 | 1/1 |
| shoves | 0/6 | 4/6 |
| other bodies and floors | 1/8 | 7/8 |
| poses never trained from | 0/4 | 4/4 |
| **total** | **2/19** | **16/19** |

Held-out: stands 12/12, full 6.00 s each, mean return 5.69 against ~6.0 for a flawless episode.
49,500 rollouts in 87 minutes.

The pose row is the clearest signal. Leaning forward or back, a deeper knee bend, a staggered
stance — the memorised policy fell within half a second of every one; this holds all four for
eight seconds. It is reacting to the body it is in.

The timestep row is the other one. Training drew from {0.5, 0.35, 0.25} ms, and the policy
holds down to 0.1 ms — four times finer than anything it saw:

| 1.0 ms | 0.8 ms | 0.6 ms | 0.5 ms | 0.35 ms | 0.25 ms | 0.15 ms | 0.1 ms |
|---|---|---|---|---|---|---|---|
| fell 0.54 s | fell 0.58 s | stood | stood | stood | stood | stood | stood |

The memorised policy fell at 0.25 ms. Refining the integrator no longer breaks this one, which
was the whole tell. It fails only going *coarser* than trained, where the contact ledger is
also at its worst (9.9 % unaccounted at 1 ms), so that failure is at least partly the
simulator rather than the policy and is not claimed as either.

The two shove failures are 120 and 160 N·s — a 1.6 and 2.1 m/s instantaneous velocity change
on a standing body. It survives 80 N·s, and at 120 N·s it staggers 3.6 m before going down.

```sh
node scripts/probe.mjs dist/balance.json
```


## Reference result

The bundled example is generated from the same code, using shared scope, seed 42, 60 generations, and a 75 kg / friction 0.90 / 0.35 m hurdle base condition that is then randomised per generation. It uses 16,336 training trials (16 to seed the champion, then 60 × (32 candidates × 2 events × 4 conditions + 2 × 8 held-out checks)). Times below are at the base condition.

| Controller | 100 m | 60 m hurdles |
|---|---:|---:|
| Hand-set baseline | 23.006 s | Fails at hurdle 1 |
| Saved shared example | 17.060 s | Fails at hurdle 1 |

These are simulation results, not predictions of human performance.

The gait is now the shape of running rather than a bounce: 0.114 s of contact, 0.109 s of flight, duty factor 0.51, 4.47 steps per second, peak ground reaction 5.07 body weights. Human sprinting sits near 0.09–0.11 s of contact, duty 0.45–0.50, and 4.5–5 steps per second. Before the limb existed the same lab produced duty 0.37 at 3.5 steps per second — a bounding gait, because nothing charged for swinging a leg.

The sprint remains a property of the model rather than the timestep: 17.060 s, 17.056 s, 17.044 s, 17.038 s at 2 ms, 1 ms, 0.5 ms and 0.25 ms. A 60-generation sprint specialist reaches 17.018 s and finishes all eight sensitivity conditions.

**Two things got harder, and both were previously being paid for by the model rather than the controller.**

The hurdle height had to come down from 0.60 m to 0.35 m. Retraction used to shrink a clearance envelope for free; it is now a leg-length change that costs reach, and clearance is checked against the limb's actual geometry. At 0.60 m the course is impassable for this body, and 0.45 m is marginal — a hurdles specialist misses hurdle 2 by 12 mm. At 0.35 m a specialist clears all five in 16.810 s and survives five of the eight sensitivity conditions.

**The shared controller still does not clear the hurdle course**, and now fails at hurdle 1 rather than hurdle 5. The held-out score plateaus at 191 by generation 50 and is unchanged through generation 150, so this is not a search-budget problem: CEM settles into the sprint-only basin, and eight parameters with one hand-designed takeoff reflex have no way to phase-lock onto five fixed hurdles. Fixing that needs a different controller, not more trials.

### What the earlier numbers were

| Version | Baseline 100 m | Example 100 m | Example hurdles |
|---|---:|---:|---:|
| Original | 22.486 s | 16.732 s | 11.092 s, all cleared |
| Contact and ledger fixed | 25.814 s | 18.990 s | Fails at hurdle 5 |
| Limbed model (current) | 23.006 s | 17.060 s | Fails at hurdle 1 |

The original figures came from a contact model that created about 381 J of spring energy per 100 m — roughly 9 % of actuator work — because a stance could open with the leg already 6–11 cm compressed whenever the body apexed below its leg-reach height, and CEM was selecting for it. That version's hurdle result existed only at its 2 ms training timestep; at 1 ms and 0.5 ms the same policy hit hurdle 1.

Fixing contact exposed a second exploit of the same species. The low-apex fallback placed the foot at the reachable distance, which is a teleport. Once the limb had mass and a bounded hip, the controller learned to ignore it entirely and scuttle: 98 % of contacts low-apex, duty factor 0.81, ten steps per second, and the planted foot 0.6–0.8 m from where the controller asked for it. Making the limb flex to fit instead of teleporting removed that, and the realistic gait above is what replaced it.
