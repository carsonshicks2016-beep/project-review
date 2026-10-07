# Faithful Porsche 919 Evo Record Program

`faithful-v2` is a new program, not a continuation of Fable Five. Its goal is
to determine—without manufacturing pace—whether a driver-equivalent policy can
complete the official 20.832 km T13-to-T13 Nordschleife flying lap in less than
**319.546 seconds**.

The retired July 2026 experiment is archived as
`legacy_misstamped_noncertifiable`. It physically selected the Porsche preset
but carried a `mazda787b-5spd-ring-v1` drivetrain identity and legacy
observation, optimizer, HOF, sector and evaluation state. It may be replayed as
history. It may never be resumed, adapted, evaluated, or promoted as
`faithful-v2` evidence.

## What can be claimed today

The repository now contains the versioned schemas, a signed external evidence
vault and holdout audit, evidence readiness checks, a pinned Linux runtime
contract, a runnable noncertifying MuJoCo baseline, fail-closed twin/oracle
interfaces, record identity and artifact isolation, adversarial
lap/certificate gates, the driver-interface boundary and recurrent network, an
original geometry package, and telemetry-to-A/V adapters.

It does **not** contain licensed Michelin tyre data, Porsche mass-property,
suspension, aero, powertrain or embedded-controller maps, June 2018 survey
data, or calibration/holdout telemetry. Nor does it yet contain the complete
licensed authority model, correlated vectorized MJX twin, CasADi/IPOPT
collocation plus nonlinear-MPC execution layer, faithful audio DSP runtime, or
end-to-end training/QMC/certification runner. The current result is therefore a
**telemetry-constrained approximation scaffold**. `TRAINING`, `PHYSICS
VALIDATED`, `ORACLE FEASIBLE`, `SIM RECORD`, and `ROBUST RECORD` must remain
false until their implementations, evidence, and numerical tests pass.

No amount of PPO training can clear a missing-evidence gate.

The evidence signature verifier is implemented, but the checked-in trust store
contains no custodian key and no licensed freeze is bound to the run. The
surveyed timing-plane/boundary/tyre-contact verifier, CasADi/MPC execution
verifier, and action-only replay/QMC certification runner remain absent. Hash
strings, caller-supplied pass booleans, fixture packages, a kinematically
plausible straight-line trace, or a self-signed test bundle therefore remain
noncertifying even when their arithmetic is below the benchmark.

## Current operator commands

The CLI exposes the new program without routing through legacy Fable artifacts:

```bash
# Computed JSON status; every public-bootstrap gate is currently false
python3 run.py --faithful-919-status

# Run schema, backend, record, policy, A/V, and control-plane validators
python3 run.py --faithful-919-validate

# Reserve a namespace only; this creates no identity or evidence claim
python3 run.py --faithful-919-init record-program-0001

# These are preflights and currently exit 78 before launching any work
python3 run.py --faithful-919-oracle --faithful-run-id record-program-0001
python3 run.py --faithful-919-train 1000 --faithful-run-id record-program-0001
```

Evidence-vault operations use an absolute external path selected by
`FAITHFUL_EVIDENCE_ROOT` or `--faithful-evidence-root`:

```bash
python3 run.py --faithful-evidence-inventory
python3 run.py --faithful-evidence-ingest /secure/incoming/package-v1
python3 run.py --faithful-evidence-verify package-v1
python3 run.py --faithful-evidence-freeze /secure/requests/freeze-v1.json \
  --faithful-evidence-signing-key /secure/keys/custodian.key
python3 run.py --faithful-evidence-open-holdout freeze-v1 \
  --faithful-evidence-actor calibration-custodian \
  --faithful-evidence-signing-key /secure/keys/custodian.key
```

Run skeletons live only under
`runtime/fable5/editions/porsche-919evo-faithful-v2/runs/<run-id>/` and are
explicitly noncertifying. Status ignores caller-authored gate booleans; no
manifest can authorize physics, oracle, training, or a record claim without a
future cryptographic verifier for the corresponding materialized artifacts.

## Frozen contract

- Timing: full 20.832 km T13-to-T13 flying lap, with interpolated timing-plane
  crossings and persistent outlap state.
- Benchmark: strictly less than 319.546 seconds.
- Vehicle operating mass: 849 kg vehicle plus 39 kg driver/ballast plus fuel;
  849 kg is not treated as total running mass.
- Driver loop: 50 Hz. Physics and validated embedded-control loops: 1 kHz.
- Authority: deterministic double-precision MuJoCo at 1 ms.
- Training twin: MJX derived from the same MJCF, parameter registry and force
  laws, with authority sampling for out-of-distribution states.
- Offline ceiling: whole-lap CasADi/IPOPT plan followed by nonlinear-MPC replay
  through the authority.
- Driver boundary: onboard-measurable signals and static map preview only.
  Simulator truth may be used by a privileged critic during training but never
  by the actor at inference.
- Undocumented boost, DRS, ABS, torque-vectoring or vehicle-level controls are
  disabled. A licensed 2018 driver/control-interface source is required to
  enable each one.

Public anchors are preserved with source identity from the
[Nürburgring record definition](https://nuerburgring.de/info/nuerburgring/records?locale=en),
[Porsche technical specification](https://newsroom.porsche.com/en/motorsports/porsche-919-hybrid-evo-top-5-series-technical-check-16834.html),
[Porsche record release](https://newsroom.porsche.com/en/motorsports/porsche-919-hybrid-evo-record-nuerburgring-nordschleife-5-minutes-19-seconds-55-timo-bernhard-15752.html),
and [Michelin's simulation correlation study](https://simulation.michelin.com/canopy/technical-articles/f1-vs-porsche-919-evo-at-the-nordschleife).
Public anchors are necessary but are not substitutes for the licensed inputs.

## Program gates

| State | Passing evidence |
|---|---|
| `TRAINING` | A run has an immutable identity, actor boundary audit passes, and it uses only the training twin associated with that identity. This state is not a record claim. |
| `PHYSICS VALIDATED` | Component, Spa calibration, frozen Nord holdout, energy and numerical tests all pass inside their stated uncertainty. No high-sensitivity parameter is unbounded. |
| `ORACLE FEASIBLE` | `oracle_time + numerical_error + one_sided_95_percent_model_penalty < 319.546`, after authority/MPC replay agrees with collocation within 0.2 s and all constraints pass. |
| `SIM RECORD` | A driver-equivalent action-only replay completes one legal continuous flying lap below 319.546 s and reproduces the telemetry hash. |
| `ROBUST RECORD` | 1,000 frozen-posterior quasi-Monte Carlo samples are at least 99% valid and the one-sided 95% bootstrap upper bound on the 95th-percentile time is below 319.546 s. |

The first failed mandatory gate stops the downstream program. In particular,
if the validated oracle gate fails, record-policy training must not start. The
output is then a sector-loss, setup-sensitivity and uncertainty-attribution
report—not a faster fictional car.

## Evidence and parameter rules

Every physical parameter is immutable and carries:

- value or probability distribution and SI units;
- source URI and content hash;
- licence and redistribution policy;
- confidence: `official`, `licensed-measured`, `calibrated`, or `inferred`;
- calibration dataset identity.

The evidence evaluator blocks the `faithful` label when a required evidence
class is missing, untrusted, hash-mismatched, unlicensed, or when a
high-sensitivity parameter has no bounded distribution. Raw licensed material
stays in an access-controlled directory outside Git. Only schemas, adapters,
permitted derived values and source hashes belong here. Nordschleife record
telemetry is a holdout: parameter calibration is frozen before its encryption
key/access is opened.

Required evidence classes are:

1. mass properties, suspension/K&C and pitch-link behavior;
2. Michelin combined-slip, load, camber, pressure, speed, temperature, wear,
   relaxation and vertical-stiffness behavior;
3. aero maps and actuator/interlock behavior;
4. ICE, MGU, battery, gearbox, losses, brake-by-wire and control maps;
5. June 2018 3D surface, legal boundaries, curbs, barriers and materials;
6. Spa calibration telemetry and Nordschleife holdout telemetry, including
   controls, line, energy, temperatures and weather.

The machine-readable acquisition matrix lists exact protocol parameters,
units, sensitivity flags, survey tolerances and validation thresholds:

```bash
python3 tools/faithful_evidence_requirements.py
```

### Custodian and package workflow

The protocol trust store is
`supra/faithful/trusted_evidence_keys.json`. It contains public keys only and is
empty by default. A data custodian creates keys outside the repository:

```bash
python3 tools/faithful_evidence_custodian.py generate-key \
  --private-key /secure/keys/custodian.key \
  --public-key /secure/keys/custodian.pub \
  --trust-entry /secure/keys/custodian-trust-entry.json \
  --label "919 evidence custodian"
```

The public trust entry is reviewed before being copied into the checked-in
trust store. The private seed is never placed in Git or printed. The custodian
then prepares an external package directory plus an external
`faithful-evidence-package-request-v1` JSON file and signs it with:

```bash
python3 tools/faithful_evidence_custodian.py sign-package \
  --package-root /secure/incoming/package-v1 \
  --request /secure/requests/package-v1.json \
  --private-key /secure/keys/custodian.key
```

Copy `supra/faithful/evidence_package_request.example.json` as the starting
request. It enumerates every required role and is deliberately labeled
`fixture`; a licensed custodian must replace the paths, source/licence metadata
and package kind from the actual data agreement rather than relabeling the
example files.

The tool computes every size and digest from the actual files. Ingestion
rejects unlisted files, symlinks, changed bytes, unknown/revoked signers,
duplicate package IDs and fixture-licence laundering. A freeze requires exactly
one source for all protocol-owned evidence roles and binds the package manifest
hashes plus vehicle, track, scenario and parameter-registry hashes.

Nord telemetry must be packaged with the dedicated holdout role and sealed
flag. Opening it appends one signed, hash-chained event for a verified freeze;
it does not silently mutate the freeze. A `fixture` package remains
noncertifying permanently, even when signed and role-complete.

### Pinned authority runtime

The authority/oracle dependency set is exact-pinned in
`requirements-faithful-lock.txt`. The Linux/amd64 container uses a digest-pinned
Python 3.12.10 Bookworm image and runs MuJoCo 1 ms, JAX x64 and IPOPT smoke
tests during its build:

```bash
docker buildx build --platform linux/amd64 --load \
  -t supra-faithful-runtime:dev \
  -f containers/faithful/Dockerfile .
docker run --rm --platform linux/amd64 supra-faithful-runtime:dev
```

The same hash-locked packages pass in an isolated Python 3.12 Mac environment.
The Linux/amd64 development image has also built and produced a materialized
MuJoCo/JAX/IPOPT probe under `runtime/faithful/`. Status reports that
development probe separately from certification: it remains noncertifying
until the exact image and probe are independently signed and bound to
`RunIdentity`.

## Physical architecture

`VehicleSpecV2`, `TrackSurfaceV2` and `RecordScenarioV1` are the immutable
physical inputs. `PhysicsIdentity` binds the authoritative solver, timestep,
model/source/dependency hashes and numerical settings. `MuJoCoAuthority` and
`MJXTrainingTwin` implement the same `PhysicsBackend` lifecycle: `reset`,
`step`, and `snapshot`.

The complete licensed authority model must include:

- six chassis DOFs, four unsprung DOFs, wheel spin and steering dynamics;
- pushrod-equivalent kinematics, stops, heave/roll coupling and pitch link;
- combined-slip tyres with aligning moment and thermal/pressure/wear state;
- separate rear ICE and front MGU with energy-conserving battery, KERS,
  exhaust recovery, SOC/current/temperature limits and physical derating;
- hydraulic/carbon brakes, regenerative blending and sourced brake-by-wire;
- ride-height/pitch/roll/yaw/element aero maps applied at measured centers;
- fuel-dependent mass/CG, physical contact, tyre lift, flight and landing;
- one canonical body/wheel geometry for collision, legal footprint, rendering
  and audio source placement.

The existing OSM/DEM constant-width Nordschleife is explicitly a cheap training
fallback. Certification requires agreement with the frozen 2018 survey inside
0.1 m lap length, 50 mm boundary position, 10 mm surface height and 0.1 degree
bank.

## Oracle, training and certification flow

1. Validate components and calibrate against component/rig and Spa data.
2. Freeze parameters and source hashes before opening the Nord holdout.
3. Validate the holdout, energy ledger and 1 ms versus 0.5 ms numerics.
4. Optimize racing line, speed, pedals, gear, hybrid/regen, SOC, brake
   distribution, sourced aero controls and thermal states as a whole lap.
5. Replay with nonlinear MPC until plan/authority time differs by less than
   0.2 s and no constraint is violated.
6. Apply numerical and one-sided 95% model-uncertainty penalties. Stop if the
   resulting bound is not below 319.546 s.
7. Train the recurrent driver by oracle/MPC cloning, DAgger recovery and PPO
   with a privileged critic. Randomize only over the frozen evidence posterior.
8. Transfer every candidate to authority evaluation; surrogate-only times are
   never record results.
9. Certify a continuous warm-up/outlap plus flying lap, then perform action-only
   deterministic replay and the 1,000-sample robustness protocol.

The actor is two 256-unit feature layers feeding a 128-unit GRU. It outputs a
squashed-Gaussian steering command, independently bounded accelerator and
brake, and categorical downshift/hold/upshift. The validated ECU owns traction,
brake-by-wire, torque distribution, hybrid safety, pitch-link and aero
interlocks.

## Artifact and tamper boundary

Faithful artifacts live under:

```text
runtime/fable5/editions/<edition>/runs/<run-id>/
```

Every run owns its manifest, events, latest training evaluation, pit state,
stage HOFs, full-lap best, oracle result and certificates. An edition owns only
same-identity certified champions. `RunIdentity` binds edition/run ID plus
vehicle, track, physics, controller, observation, action, sources,
dependencies and protocol hashes.

Exact resume requires all identity hashes plus optimizer, scheduler, RNG,
normalizer and budget state. A stage warm start resets optimizer and evaluation
evidence. Cross-car or cross-physics migration requires an explicit adapter and
can never carry certification. Any change to the policy, source, dependency,
solver, scenario, parameter, controller, action trace or verification protocol
invalidates a certificate.

## Geometry and audio truth

The checked-in Porsche asset is an original private-research interpretation at
the official 5.078 x 1.900 x 1.050 m envelope. Its provenance manifest and
named nodes establish the candidate canonical frame for render/audio adapters.
Certification collision and 2D-footprint derivation are not wired to this
asset yet. It is not a claim of licensed measured surface accuracy; geometry
certification remains blocked until that evidence is supplied and the shared
geometry path is implemented and validated.

The checked-in `TruthfulAvAdapter` permits visual suspension, wheel rotation,
brake glow and active elements only when the corresponding
`VehicleTelemetryV2` channels exist. Its audio projection likewise accepts V4
combustion state from crank torque/RPM, turbo state, signed MGU power, regen
only from negative MGU power, gearbox shaft/shift state, and independent tyre
slip/load/contact. The faithful runtime DSP and full telemetry wiring are not
implemented yet, so legacy Porsche viewer audio is disabled and unsupported
animation freezes. Missing telemetry produces silence or a visible warning,
never a Mazda/Supra fallback or invented boost/regen.

## Acceptance thresholds

- Energy residual below 0.1% for the full authority lap.
- Spa lap inside 0.5%, speed anchors inside 1%, lateral acceleration inside
  0.25 g.
- Frozen Nord replay inside 0.5%, peak/mean speed inside 1%, normalized speed
  trace RMSE below 5%.
- 0.5 ms versus 1 ms authority time delta below 0.1 s.
- MJX standardized trajectories inside 1% force, 0.5 m/s speed and 0.5 m
  position of MuJoCo.
- Five deterministic certificate replays and an independent action-only replay
  agree within 1 ms and reproduce the telemetry hash.
- A/V output is deterministic; no false hybrid events; callback p99 below 25%
  of deadline; ten-minute target-Mac replay has no underruns.

A passing bundle supports the phrase **robust faithful-simulation record**. It
does not claim that a physical Porsche has set a new real-world record; that
would require an instrumented real-car track test.
