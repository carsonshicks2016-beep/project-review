# PHYSICS 3D PLAN — Elevation, Gravity, Jumps & Pitch-Aware Agents

**REVISION 2 (2026-06-10).** Supersedes rev 1 of the same day. Owner directive:
hills must be *fully* physical — per-track gravity along incline/decline,
correct momentum everywhere, and **jumps**: hit a crest too fast and the car
launches like a ramp, flies ballistically, and lands. Agents get an expanded
sensor block (pitch, vertical state, crest preview) so they can be trained on
the real 3D physics, and the Command Center becomes the training environment
for it. Rev 1's "car glued to road / no jumps" lock (old Decision #1) is
**overturned by the owner** — airtime is in scope for hills-v1.

**Standing decision (owner, 2026-06-09, reaffirmed 2026-06-10):** breaking ALL
existing PPO/GA checkpoints is accepted. We extend the obs vector and retrain.
The invariant we keep is *physics byte-identity on flat ground* — proven by
regression gates, so the refactor itself adds zero numerical drift, and a flat
track can never trigger the airborne path.

This is Phase 8 of `VIEWER3D_V2_PLAN.md`, promoted to its own doc because it
touches: `supra/track.py`, `supra/physics.py`, `supra/sensors.py`,
`supra/config.py`, `supra/ppo_env.py`, `supra/ppo.py`, `supra/evolution.py`,
`supra/agent.py`, `supra/app.py`, `supra/dashboard.py`, `run.py`,
`viewer3d/session.py`, `viewer3d/static/` (v2), `command-center/server.py` +
UI.

---

## 0. Architecture of the change (read this before any stage)

### 0.1 The one mental model

A track stays a 2D plan (centerline x,y) **plus a height field along arc
length**: `z(s)`, its slope `grade(s) = dz/ds` (dimensionless, + = climb), its
vertical curvature `vcurv(s) = d(grade)/ds` (1/m, **negative at crests,
positive in dips** — fix the sign convention here once, rev 1 had it flipped),
and a lateral tilt `bank(s)` (rad, + = left edge higher).

The car is a planar dynamics model (x, y, yaw) **plus a vertical degree of
freedom (z, vz) with two modes**:

- **GROUNDED** (the constraint mode): the car follows the road surface.
  Gravity's in-plane component is a real body force (climbs pull back,
  descents push, banks pull sideways), normal load scales with slope and with
  v²·vcurv (light over crests, heavy in dips), and pitch/roll are kinematic
  outputs from the road plane. `z = road_z`, `vz = speed·grade`.
- **AIRBORNE** (the ballistic mode): when the crest demands more downward
  acceleration than gravity supplies (`v² ≥ g/|vcurv|` at a crest, or the
  road drops out from under a ramp edge), the wheels unload to zero and the
  car becomes a projectile. Planar momentum is conserved (only aero drag acts
  in-plane — **this is the "momentum works correctly" requirement**), vz
  integrates under −g, the tyres produce nothing, steering does nothing, the
  engine free-revs. Landing re-grounds the car through a suspension impact
  model that spikes the loads and (via the existing load-sensitive friction)
  briefly costs grip — hard landings punish themselves physically, no reward
  hacks needed.

Takeoff is **emergent, not scripted**: it falls out of the normal-load
equation reaching zero. There is no "jump zone" data — any crest jumped fast
enough is a ramp, exactly as asked.

### 0.2 Data-flow contract (mirrors the existing `surface_grip` pattern)

Physics never reads the track. Every driving loop already does, per step:

```python
fr = trk.frame(veh.x, veh.y)
veh.surface_grip = spec.offtrack_grip if fr["off_track"] else 1.0
```
(call sites: `supra/app.py`, `supra/agent.py`, `supra/ppo_env.py` step+reset,
`viewer3d/session.py` `_step_fixed`)

We extend `frame()` to also return `z`, `grade`, `bank`, `vcurv`, and add ONE
line at each call site:

```python
veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
```

`Vehicle.set_road` projects the track-frame slope onto the car's body axes
using `yaw - heading` (a car sliding sideways down a climb feels the pull on
its *side*). While AIRBORNE, `set_road` only refreshes the road height under
the car (for landing detection) — slope forces are suspended because a free
body in flight feels no surface. Defaults are zero → any caller that never
calls `set_road` (old code, tools) gets exact current behavior, permanently
grounded on flat.

### 0.3 Byte-identity rules (how the physics stages stay provably safe)

- All new force/load terms are **unconditional additions of terms that are
  exactly `0.0` when grade=bank=vcurv=0** (`x + 0.0 == x` in IEEE754; no
  branches that change instruction order on the flat path).
- The airborne branch is gated on a condition (`normal-load factor ≤ 0`) that
  is **unsatisfiable on a flat track** (factor ≡ 1.0). The grounded code path
  computes identically whether the branch exists or not.
- No reordering of existing arithmetic. New terms append at the end of the
  existing sums (`Fx_tot = ... - rr_x + self._grav_fx`).
- Gate: `tools/regression_baseline.py` (Stage 0) records checksums BEFORE any
  edit; rerun after every stage. Pattern: `tools/regression_drivetrain.py`.

### 0.4 Observation layout (the breaking change, made once)

Current vector (40): `9 beams + 25 proprio + 6 lookahead-curvature`
(`sensors.py: obs_size = n_beams + 25 + len(lookahead_distances)`); env
appends a 2-dim mode one-hot → `obs_dim 42`.

New vector (58): the old 40 **unchanged and in the same order**, then an
appended **hill/air block (18)** — this is the "more sensors for pitch data":

```
[40] grade_body        / 0.20   # road slope along heading (+ = climbing), ~±11° ref
[41] bank_body         / 0.10   # road lateral tilt under the car (+ = left up)
[42] pitch             / 0.30   # ACTUAL body pitch (rad) — = road pitch grounded,
                                #   ballistic attitude in the air
[43] vz                / 12.0   # vertical velocity (m/s) — launch/landing anticipation
[44] height_above_road / 3.0    # 0 grounded; >0 flying (landing timer the net can learn)
[45] airborne                   # 0 ground / 1 air (hard contact flag — "you have no grip")
[46..51] lookahead_grade at the SAME 6 distances as curvature / 0.20
[52..57] lookahead_vcurv at the SAME 6 distances / 0.02
                                # crest/dip preview — the "jump detector": together
                                #   with own speed the net can predict takeoff
```

→ `obs_size 58`, `obs_dim 60`. Appending (not interleaving) keeps every
existing index/label stable — dashboard beam/proprio/curvature rendering and
`proprio_labels` stay valid; only new labels are added.

Why these 18: the agent must *anticipate* hills exactly like it anticipates
curvature (brake earlier into a downhill hairpin, carry speed into a climb),
AND it must reason about flight: `lookahead_vcurv × speed` predicts takeoff
before it happens, `vz + height_above_road` times the landing, `airborne`
says "steering and throttle do nothing right now", `pitch` is the body
attitude it will land on. Same 6 distances as curvature → "downhill hairpin"
and "jumpable crest" each read as one joint signature.

### 0.5 Checkpoint fallout (accepted; make it LOUD, not silent)

- **PPO**: `ppo.py` already raises on `obs_dim` mismatch — old `.pt` files
  (obs 42) fail with a clear message. Improve the message: append "(pre-hills
  checkpoint — retrain; see PHYSICS_3D_PLAN.md)". Save dict gains
  `"obs_layout": "hills-v1"`.
- **GA**: `evolution.py` stores `obs_size` in the npz but `from_genome` is
  called with the CURRENT obs_size → garbage reshape risk. Add an explicit
  check: stored vs current, raise the same loud error.
- **Command Center**: watch/resume of an old checkpoint must surface the
  server-side error string in the UI (it streams stderr — verify), and the
  checkpoint browser badges pre-hills files (Stage 6).
- Existing `.pt`/`.npz` files on disk stay untouched (museum pieces; the
  loader just refuses them).

### 0.6 What deliberately does NOT change

- **Reward functions** (progress is 2D arc length; speed/slip terms are
  slope-agnostic by construction). Jumps need no reward engineering in v1:
  flying off-track still eats the positional off-track penalty (discourages
  corner-cutting by air), hard landings cost grip and therefore time, and
  airtime generates no extra progress. Two surgical gates only (Stage 5):
  drift/hybrid STYLE bonuses require `not airborne` (no credit for "drifting"
  in the air), and that's it. Retune only if Stage 7 evals show an exploit.
- Action space, control rate, AutoBox, sound synthesis (engine load shifts on
  grades and the free-rev in air emerge naturally from rpm/throttle).
- The 2D PyGame viewer stays top-down (grade% + AIR indicator added to the
  dashboard so hills/jumps are *legible*, Stage 8).
- `velocity` integration scheme, dt, substeps.

---

## 1. Stage 0 — Baseline harness (BEFORE any edit)

New `tools/regression_baseline.py`:
1. The existing drivetrain run (verbatim from `regression_drivetrain.py`) for
   supra/rx7/skyline → checksums.
2. A **track-coupled scripted lap**: car + `named_track("club")` and
   `named_track("akina")`, the `agent.py`-style loop (frame → surface_grip →
   scripted controls 3000 steps) → trajectory checksum per track.
3. A **sensor checksum**: `SensorSuite.observe` vector sums along those runs
   (locks obs semantics for the old 40 dims).
4. Writes/compares `tools/baselines_prehills.json` (`--write` vs default
   compare mode, exits non-zero on mismatch).
5. From Stage 3 on, compare mode ALSO asserts `airborne_steps == 0` across
   every baseline run (flat ⇒ flight is impossible — cheap canary for a
   takeoff-condition bug).

**Gate 0:** baselines file committed; compare mode passes trivially.
Run with `PYTHONPATH="$PWD"` per house convention.

## 2. Stage 1 — Track height field + profile generator (zeros by default)

`supra/track.py`:
- `Track.__init__(centerline, width=12.0, elevation=None, bank=None)`;
  `_build()` adds:
  ```python
  self.z = elevation if ... else np.zeros(M)
  self.bank = bank if ... else np.zeros(M)
  # periodic central differences along arc (same scheme as _curvature):
  self.grade = dz/ds          # + = climbing
  self.vcurv = d(grade)/ds    # NEGATIVE at crests, POSITIVE in dips (§0.1)
  self.elev_gain = float(z.max() - z.min())
  # per-point launch speed: v at which a crest stops holding the car down
  self.launch_speed = np.where(self.vcurv < -1e-6,
                               np.sqrt(G / -self.vcurv), np.inf)
  ```
- `frame()` adds keys: `z`, `grade`, `bank`, `vcurv` (values at index i).
- `lookahead_grade(arc, distances)` and `lookahead_vcurv(arc, distances)` —
  clones of `lookahead_curvature`.
- `pose_at`/`start_pose` unchanged (x, y, yaw); callers get z via frame.

**The profile generator** `_elevation_profile(track, style, difficulty, rng)`:
- Loop-periodic by construction: `z(s) = Σ_k A_k · sin(2π·k·s/L + φ_k)`,
  k ∈ {1,2,3,5,8}, amplitudes drawn per style then shaped:
  | style     | total amp (easy→hard) | max grade cap |
  |-----------|----------------------|---------------|
  | touge     | 8 → 26 m             | 0.14          |
  | technical | 5 → 14 m             | 0.10          |
  | gp        | 4 → 10 m             | 0.07          |
  | speedway  | 0 → 3 m              | 0.03          |
- Plus a curvature-correlated term (touge/technical only): smoothed
  `|curvature|` integrated → hairpins sit near local crests/saddles like real
  switchback stacks (weight ~25% of amplitude).
- **Constraint pass:** compute grade; if `max|grade| > cap`, scale z by
  `cap/max|grade|`. Then smoothing on z — but now **vcurv-aware, not blanket**:
  smooth until every crest's `launch_speed` ≥ a per-style floor, so jumps are
  *possible but honest* (no flying at parking-lot speed):
  | style     | min launch speed (easy→hard) |
  |-----------|------------------------------|
  | touge     | 38 → 26 m/s  (~137→94 km/h)  |
  | technical | 42 → 32 m/s                  |
  | gp        | 55 → 45 m/s                  |
  | speedway  | ∞ (no jumpable crests)       |
  Hard touge therefore has crests a committed agent (or owner) WILL clear —
  that's the feature, sized so it's a choice, not an ambush.
- **Named showcase track**: add `"ridge"` to the named set — touge-hard with
  two authored jumpable crests (one straight launch, one cresting into a
  bend). Seeded via `_stable_seed("ridge")` like every other named track;
  becomes the jump eval/showcase venue.
- Banking: data path ships now, generator keeps `bank ≡ 0` in hills-v1
  (one variable at a time for retraining). A `bank_corners(track, max_deg)`
  helper is written + unit-tested but not wired into generators (Stage 7+
  option for gp/speedway).
- Wiring: `make_track(..., hills=True, hill_scale=1.0)` and
  `stadium(..., hills=True)` / `_handcrafted(...)` compute the profile from
  their style/difficulty with their existing seeds (named tracks therefore get
  deterministic hills via `_stable_seed(name)` — train/eval/watch/thumbnail
  all agree, same as layouts). **Default `hills=False` until Stage 4 flips
  it** — Stages 1–3 land with zero behavior change.

**Gate 1:** baseline compare passes byte-identical (hills off everywhere);
new `tools/validate_track_elevation.py` checks: periodicity (z wraps
seamlessly), grade caps honored per style, launch-speed floors honored,
`ridge` has exactly its two designed jumpable crests, determinism for named
tracks across two processes.

## 3. Stage 2 — Grounded 2.5D physics (zero-input = identical)

`supra/physics.py` (`Vehicle`):
- New state (set in `reset`): `self.grade_body = 0.0; self.bank_body = 0.0;
  self.road_z = 0.0; self.road_vcurv = 0.0; self.z = 0.0; self.vz = 0.0;
  self.airborne = False; self.air_time = 0.0; self.landing_g = 0.0;
  self.pitch = 0.0; self.roll = 0.0; self._contact = 1.0`.
- `set_road(self, grade, bank, heading, z, vcurv=0.0)`:
  ```python
  rel = yaw - heading                      # car vs track tangent
  c, s = cos(rel), sin(rel)
  self.road_z = z; self.road_vcurv = vcurv
  if not self.airborne:
      self.grade_body =  grade * c - bank * s  # slope under the nose
      self.bank_body  =  grade * s + bank * c  # slope under the left side
      self.z = z; self.vz = self.vx * self.grade_body
      self.pitch = atan(grade_body); self.roll = atan(bank_body)  # outputs
  # airborne: only road_z/road_vcurv refresh (landing detection); slopes,
  # z, vz, pitch are owned by the flight integrator (Stage 3)
  ```
  (small-angle treatment of the slope vector rotation; exact enough ≤ 0.2.)
- In `step()` (after the existing force sums, appended terms):
  ```python
  m = s.mass
  self._grav_fx = -m * G * self.grade_body      # pulls back on a climb
  self._grav_fy = -m * G * self.bank_body       # pulls toward the low side
  Fx_tot = float(Fbx.sum()) - drag_x - rr_x + self._grav_fx
  Fy_tot = float(Fby.sum()) - drag_y - rr_y + self._grav_fy
  ```
  NOTE: `self.ax/ay` then INCLUDE the gravity pull — that is what the
  accelerometer-style obs should feel, and what ATTESA `g_need` should see.
- In `_update_loads()` (appended, all zero on flat):
  ```python
  slope2 = grade_body**2 + bank_body**2
  cosN = 1.0 / np.sqrt(1.0 + slope2)            # normal-load factor
  # v² crest/dip term — vcurv NEGATIVE at crests (§0.1 convention):
  vert_raw = 1.0 + self.road_vcurv * self.vx * self.vx / G
  vert = np.clip(vert_raw, 0.0, 1.7)            # floor 0: crests can FULLY unload
  scaleN = cosN * vert
  target *= scaleN
  # static transfer from the slope (climb loads the rear; left-up loads right)
  gt = s.mass * G * self.grade_body * s.cg_height / s.wheelbase
  target[FRONT] -= gt / 2.0;  target[REAR] += gt / 2.0
  bt = s.mass * G * self.bank_body * s.cg_height / s.track_width
  btf = bt * s.roll_front_frac; btr = bt * (1.0 - s.roll_front_frac)
  target[FL] -= btf; target[FR] += btf; target[RL] -= btr; target[RR] += btr
  target = np.maximum(target, 0.0)
  self._vert_raw = vert_raw                     # Stage 3 reads this for takeoff
  ```
  CRITICAL transfer-sign note: the slope transfer must NOT route through the
  existing `self.ax`-based term — gravity-in-`ax` would shift load the wrong
  way (braking-style). Keep it a separate explicit term as above; the
  existing dynamic-transfer lines stay untouched.
- `telemetry()` gains `pitch`, `roll`, `grade_body`, `bank_body`, `z`, `vz`,
  `airborne`, `landing_g`.

**Gate 2:** (a) baseline compare byte-identical (all roads still flat);
(b) new `tools/validate_hills.py` on a constant-grade synthetic track:
  - coast-down: Δ(½v²) ≈ g·Δh − losses (energy bookkeeping within ~5%),
  - steady climb at part throttle slows / downhill runs away,
  - 10% grade static loads: rear gain ≈ `m·g·0.1·h/L` (analytic match),
  - crest at speed: ΣFz dips by ≈ `m·v²·|vcurv|`, reaches exactly 0 at
    `v = launch_speed` (the Stage 3 trigger point, verified before the
    airborne code even exists),
  - banked corner (physics-level, bank set directly): higher steady lateral
    speed than flat for same radius.

## 4. Stage 3 — Airborne: takeoff, flight, landing  *(the jumps)*

All inside `supra/physics.py`; no track or call-site changes. New
`CarSpec`/`SimSpec` knobs: `landing_tau` (default 0.12 s — suspension impact
absorption), `air_pitch_rate` (rad/s, how fast the nose follows the flight
arc), `max_landing_load` (clamp, ×static).

**Takeoff** (checked in `step()` after `_update_loads`, grounded only):
```python
if not self.airborne and self._vert_raw <= 0.0:
    self.airborne = True; self.air_time = 0.0
    self.vz = self.vx * self.grade_body      # ramp angle sets launch vz —
                                             # "hit it like a ramp"
```
Unsatisfiable on flat (`_vert_raw ≡ 1.0`) — byte-identity holds. No
hysteresis needed for chatter: once airborne, the only way back is contact.

**Flight** (each `step()` while airborne):
- Friction budget zeroed: `self._contact = 0.0`, `D *= self._contact`, and
  rolling resistance `rr_mag *= self._contact` → tyres and rr produce exactly
  nothing; aero drag still acts (it's the only in-plane force — **planar
  momentum conserved minus drag**, the requirement). Load targets in
  `_update_loads` → 0 (suspension extends over its existing lag).
- Slope gravity terms read `grade_body = bank_body = 0.0` (free body — no
  surface, no in-plane gravity component).
- Vertical: `self.vz -= G * dt; self.z += self.vz * dt; self.air_time += dt`.
- Attitude: yaw rate `r` persists (conserved angular momentum; tyre yaw
  damping is gone because tyre forces are zero — spins thrown before takeoff
  CONTINUE in the air, which is correct and visually glorious). Pitch eases
  toward the flight-path angle `atan2(vz, speed)` at `air_pitch_rate` (nose
  follows the arc). Roll eases toward 0 slowly.
- Wheels: skip the `free_roll` ground-speed pinning while airborne (pinning
  to ground speed mid-air is nonsense); undriven wheels hold their spin with
  slight decay, driven wheels obey the drivetrain — full throttle in the air
  free-revs engine + wheels exactly like a real car off a jump (rev limiter
  already caps it; sound emerges for free).
- `kappa_lag`/`alpha_lag` relax toward 0 (no contact patch, no slip state).

**Landing** (when `self.z <= road_z` and descending relative to the road):
- Snap `z = road_z`, `airborne = False`.
- Impact: relative vertical speed `dv = (road-following vz) − self.vz` ≥ 0.
  Absorbed first-order over `landing_tau`: the load target gains a decaying
  impact term `m·dv/landing_tau`, clamped at `max_landing_load`, distributed
  front/rear by landing attitude (`pitch` vs road pitch at touchdown —
  nose-low landings hammer the fronts). `self.landing_g = dv/(G·landing_tau)`
  for telemetry/HUD.
- Grip consequences are EMERGENT, not scripted: the load spike rides through
  `_mu_effective`'s load sensitivity (overloaded tyres lose μ), and
  wheel-speed vs ground-speed mismatch on touchdown produces a kappa spike
  that the existing `clip(±4)` + relaxation length resolve — land throttle-
  pinned and sideways and you WILL be sliding, which is the correct outcome.
- No bounce/restitution in v1 (suspension absorbs all); noted as future work.

**Gate 3:** (a) baseline compare byte-identical + `airborne_steps == 0`
assertion (§ Stage 0.5); (b) `tools/validate_jumps.py` on synthetic ramp
tracks:
  - takeoff occurs within 2% of the analytic `launch_speed` of the crest,
  - launch vz within 3% of `v·grade` at the lip,
  - flight range ≈ `v²·sin(2θ)/g` (drag-corrected tolerance ~8%),
  - in-flight: ½v²+g·z conserved minus drag work (energy audit ≤ 2% drift),
    planar heading change ≡ r·t (no phantom tyre forces),
  - landing: loads settle to static within ~6·suspension_tau, no NaN, no
    negative Fz, landing_g sane vs drop height,
  - below launch speed over the same crest: never airborne, loads dip and
    recover (the Gate 2 behavior, now with the branch present).

## 5. Stage 4 — Hills ON + flags + 3D viewer integration

- Flip defaults: `make_track(hills=True)`, named/special tracks generate
  profiles. `run.py` gains `--flat` (kills elevation everywhere) and
  `--hill-scale X` (0..1.5) passed through to generators — the A/B lever and
  the nostalgia switch.
- Call sites add the `set_road` line: `app.py`, `agent.py`, `ppo_env.py`
  (step + both resets), `viewer3d/session.py` (all listed in §0.2).
- `viewer3d/session.py`: DELETE `visual_elevation`/`elevation_at_arc`;
  `track_payload` z ← `trk.z[i]` (and adds `bank` array);
  `state_payload.pose` gains `z: veh.z` (**car z, not road z** — that's what
  makes the jump visible), `pitch: veh.pitch`, `roll: veh.roll`, and the
  payload gains `airborne` + `landing_g` for HUD flair.
- v2 client (already contracted for this, VIEWER3D_V2_PLAN §2.2):
  `app.js` uses `pose.z/pitch/roll` when present (falls back to the current
  `sampleRoadAt`/`roadPitchAt` road sampling); the contact-shadow blob stays
  ON the road surface and shrinks/fades with `height_above_road` (sell the
  air); wheel spin keeps reading `wheel_w` (free-spin in air comes through
  the existing payload); `road.js` banks the ribbon from the payload's
  `bank` (visual only until banking generates non-zero).
- HUD: small AIR indicator + landing-g flash on touchdown (numbers the owner
  can feel).
- 2D thumbnails (`command-center/server.py`) are top-down → unaffected.

**Gate 4:** drive `akina` + `club` + `ridge` in the v2 viewer: hills FELT
(speed bleeds on climbs, runs on downhills, crest lightness on the grip
readouts) and SEEN (road/terrain re-drape via server z); send the `ridge`
crest at speed → car launches, flies with conserved momentum, shadow detaches,
lands with a load spike — and crawling the same crest stays planted.
`--flat` restores today's feel exactly (spot-check against baseline lap times).

## 6. Stage 5 — Sensors + training plumbing (the breaking commit)

- `config.SensorSpec`: `+ hill_block: bool = True` (escape hatch for debug),
  refs `grade_ref=0.20`, `bank_ref=0.10`, `pitch_ref=0.30`, `vz_ref=12.0`,
  `height_ref=3.0`, `vcurv_ref=0.02`.
- `sensors.py`: append the 18-dim hill/air block per §0.4 (`obs_size`
  40 → 58); labels `("grade", ...), ("bank", ...), ("pitch", ...),
  ("vz", ...), ("height", ...), ("air", ...)` added to the
  `proprio_labels`-style reporting; `Observation` keeps raw
  `lookahead_grade` + `lookahead_vcurv` arrays for viz.
- `ppo_env.py`: obs_dim → 60; gate drift/hybrid STYLE bonuses and the drift
  speed bootstrap with `not veh.airborne` (no style points for flying —
  §0.6); race reward untouched (off-track penalty already applies
  positionally during flight, progress is arc-based). Episode `info` gains
  `airtime`, `jumps`, `max_landing_g` so training logs can report them.
- `dashboard.py`: render the new bars (appended group, existing indices
  untouched) + AIR flag.
- `evolution.py`: loud obs_size check on checkpoint load (§0.5).
- `ppo.py`: improved mismatch message + `obs_layout: "hills-v1"` in the save
  dict; `[eval]` line gains `jumps=N air=X.Xs` when nonzero.
- `aiviz.py` PolicyAgent: works untouched (observe-driven) — verify.

**Gate 5:** (a) smoke training: `--ppo 5`, `--drift 5`, GA `--train 3` on
hilly curriculum: obs_dim 60 everywhere, no NaNs, rewards finite, airtime
shows up in info on `ridge`; (b) loading any pre-hills checkpoint raises the
loud message (test with an existing .pt); (c) drive with dashboard (Tab):
hill bars + AIR flag live.

## 7. Stage 6 — Command Center: the 3D training environment

`command-center/server.py` + UI:
- **Launch params**: every train/watch/continue action gains
  `hill_scale` (slider 0–1.5, default 1.0) and `flat` (checkbox) →
  `--hill-scale X` / `--flat` appended in `build_command` (same pattern as
  `_ppo_extra`). Watch viewers default to the same hills the checkpoint
  trained on (read from metadata, below).
- **Checkpoint intelligence**: metadata reader picks up `obs_layout`; the
  checkpoint browser badges files — `hills-v1` vs `pre-hills (obs 42)` —
  and pre-hills checkpoints get a warning tooltip ("incompatible — retrain")
  instead of a silent activate. Server-side stderr from a refused load is
  surfaced verbatim in the stream panel (verify the SSE path).
- **Training telemetry**: the SSE metric parser picks up the new `[eval]`
  fields (`jumps`, `air`) and the existing charts gain an airtime trace on
  hilly runs — the owner can SEE when a generation discovers jumping.
- **Track picker**: `ridge` added to the named-track lists (train target,
  watch, thumbnails). Thumbnails stay top-down 2D; optional later: shade the
  centerline by elevation (nice-to-have, not gating).
- Activation defaults move to the `*_hills` checkpoints once Stage 7
  produces them.

**Gate 6:** from the Command Center UI alone: launch a hilly PPO run with
hill_scale 0.6, watch it stream airtime metrics, stop it cleanly, see the
checkpoint badged `hills-v1`; attempt to watch a pre-hills checkpoint and get
the readable error in-panel, not a dead process.

## 8. Stage 7 — Retraining program

- Curriculum: `curriculum_track`/`drift_curriculum_track` pass
  `hill_scale = 0.4 + 0.6·difficulty` (race) / `0.25 + 0.5·difficulty`
  (drift — slopes punish slides; ramp gentler). Stadium pool stays near-flat.
  Jump exposure arrives naturally at high difficulty via the launch-speed
  floors (Stage 1 table) — no special jump curriculum in v1.
- Eval suite: the named set now implies hills; keep `--flat` eval runs as a
  secondary metric so flat-driving competence is tracked (tolerance: new
  generalist within ~10% of old `[eval]` lap metrics on flat club/national).
  `ridge` joins the eval set as the jump/commitment probe.
- Order: **race generalist** first (`--ppo`, target: curriculum difficulty
  progression matching the old run's milestones), then **drift generalist**,
  then optional specialists (akina hill-drift is the showcase; a `ridge`
  send-it specialist is the victory lap). GA champion refresh last (it's the
  cheap one).
- Naming: `*_hills` checkpoints (e.g. `generalist_hills.pt`) so the museum
  stays legible. Update Command Center activation defaults when ready.
- Expectations note: hills add variance; early curves will look worse than
  the flat runs at the same iteration — judge by `[eval]` on fixed named
  tracks only (house rule already). Watch the airtime trace: brief "jump
  everything" phases are normal exploration; sustained airtime with falling
  lap completion = the exploit check (then and only then consider a
  hard-landing reward term).

**Gate 7:** race generalist completes laps on akina-with-hills (eval lap
completion ≥ flat-era rate − 10%) and clears or correctly declines `ridge`
crests by speed; drift generalist's composite on national ≥ 80% of flat-era;
no flat-eval collapse beyond tolerance.

## 9. Stage 8 — Polish + docs

- Dashboard: grade% + pitch/roll readout + AIR/landing-g in the telemetry
  panel (2D viewer parity with the 3D HUD).
- README: physics bullets gain the 2.5D+air model; HANDOFF: short section;
  memory file update; VIEWER3D_V2_PLAN Phase 8 checked off.
- Optional (owner's call): banked-corner generation for gp/speedway
  (`bank_corners`), viewer banking visuals get real data then; landing
  bounce/restitution; suspension travel animation in the 3D viewer.

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Silent flat-path drift from refactor | Stage 0 byte-identity baselines, rerun EVERY stage |
| vcurv sign error (classic; rev 1 had it) | Convention pinned in §0.1 (crest = negative); analytic crest test in Gate 2 |
| Wrong slope-transfer sign (classic) | Analytic load check in Gate 2; explicit term, never via `ax` |
| Phantom takeoffs on flat | `_vert_raw ≡ 1.0` on flat; `airborne_steps == 0` assertion in every baseline run |
| Takeoff chatter at the threshold | One-way transition: airborne ends only on contact, never on the load factor |
| Landing force explosion / NaN | `landing_tau` floor, `max_landing_load` clamp, existing Fz≥0 + suspension lag; kappa spike bounded by existing `clip(±4)` + relaxation |
| Agents jump to cut corners | Progress is arc-based (air adds none); off-track penalty is positional and applies mid-flight; `ridge` eval watches for it; reward term held in reserve |
| Drift reward farmed mid-air | STYLE/bootstrap bonuses gated `not airborne` (Stage 5) |
| Hills destabilize early training | Curriculum hill_scale ramp; `--flat` lever; judge by `[eval]` only |
| Landing off-road where physics z ≠ viewer terrain z | Physics lands on track-frame z (clamped to nearest centerline) — authoritative; viewer draws the car at server z so no visual conflict; noted as known approximation off-road |
| Old checkpoints half-loading | Loud loader errors both trainers; Command Center badges + surfaces stderr |
| Exploring starts on steep grade (rolling start gear pick) | unchanged logic is speed-based; verify in Gate 5 smoke |
| Viewer desync (client invents z) | already contractually server-fed (v2 built that way); Stage 4 swaps the source and adds car-z |
| Perf: extra per-step work | slope math is O(1) reusing the existing `frame()` call — no new `nearest()` scans |

## 11. Decisions locked (so future sessions don't relitigate)

1. **REVISED 2026-06-10 (owner):** the car has a vertical DOF. Grounded =
   road constraint; airborne = ballistic with conserved planar momentum.
   **Jumps are in scope for hills-v1** — emergent from the load equation, no
   scripted jump zones. (Supersedes rev 1's "glued to road" lock.)
2. Obs = old 40 unchanged + 18 appended → 58 (+2 mode = 60). Append-only.
3. All pre-hills checkpoints break; loud errors; no compat shims.
4. Banking: full data path + physics now, generators emit 0 until Stage 7+.
5. Named-track hills are seed-deterministic (same `_stable_seed` discipline);
   `ridge` is the canonical jump track.
6. `--flat` stays forever as an escape hatch and A/B tool.
7. Byte-identity on flat is the non-negotiable refactor gate at every stage,
   now including the `airborne_steps == 0` canary.
8. Rewards stay un-engineered for jumps in v1 (physics self-punishes hard
   landings; positional penalties already cover air exploits); a hard-landing
   term is the named fallback if Stage 7 evals demand it.
9. vcurv convention: `d(grade)/ds`, **negative at crests** — every formula in
   this doc assumes it.

## STATUS TRACKER

- [x] Stage 0 — Baseline harness (2026-06-10: `tools/regression_baseline.py`
      + `tools/baselines_prehills.json`. Locks: drivetrain checksums
      supra/rx7/skyline; scripted-controller track runs on club (0 offtrack
      steps) + akina (1343 offtrack steps — grip transitions exercised);
      per-dim sensor sums of obs[:40] (survives the Stage 5 append, fails on
      any reorder); airborne_steps==0 canary via getattr, live from day one.
      All values stored as float hex = byte-exact. Verified: compare passes
      ×2 across processes; 1e-9 perturbation → loud FAIL exit 1. Harness
      never calls set_road, so it stays valid after Stage 4 flips hills on)
- [x] Stage 1 — Track height field + profile generator + `ridge` (2026-06-10:
      `Track` gains z/grade/vcurv/bank + per-point `launch_speed`, all built
      through one entry point `set_elevation()` (generators, `--flat`, and
      synthetic test tracks all come through it); `frame()` carries
      z/grade/bank/vcurv; `lookahead_grade`/`lookahead_vcurv` added.
      `_elevation_profile`: periodic Fourier base + curvature-correlated
      climb (touge/tech), HILLS table = per-style amp/grade-cap/launch-floor,
      vcurv-aware smoothing until crests honor the floor. Hills draw from the
      same rng stream strictly AFTER layout draws → same seed = same layout
      hills on or off; SPECIAL tracks carry _stable_seed hill seeds so the
      Stage 4 default flip stays deterministic. `ridge` = hard touge w/ two
      authored Gaussian crests (28.7 + 31.9 m/s launch, 409 m apart), sited
      from layout curvature, base locally calmed, final cap rescale.
      `bank_corners()` written + unit-tested, unwired. Defaults all OFF.
      Gate 1: validate_track_elevation.py all green (caps/floors ×4 styles
      ×3 difficulties ×3 seeds, cross-process determinism, ridge clusters,
      ctor contract); regression_baseline byte-identical; SupraEnv obs 42 +
      named tracks + viewer session smoke OK)
- [x] Stage 2 — Grounded 2.5D forces/loads + pitch/roll telemetry
      (2026-06-10: `set_road()` projects track slope onto body axes via
      yaw−heading (sideways car feels slope as BANK); slope gravity appended
      to the force sums (so ax/ay include the pull — ATTESA g_need sees it);
      loads: cosN normal factor × v²-vcurv crest/dip factor (floor 0.0 —
      `_vert_raw ≤ 0` is Stage 3's takeoff trigger) + explicit static slope
      transfer, never via ax; pitch/roll/z/vz/airborne/landing_g in
      telemetry. Gate 2: baseline byte-identical; validate_hills.py all
      green — instantaneous slope pull = −g·grade exact, coast-down energy
      audit 0.6% residual, analytic static loads to 1e-6, crest unload hits
      exactly 0 at launch speed + clamps, bank pull + load shift signs
      verified, banked corner sustains +0.93 m/s² over flat)
- [x] Stage 3 — Airborne model: takeoff / flight / landing (2026-06-11:
      `_update_vertical()` at the end of step() owns the mode transitions —
      takeoff emergent from `_vert_raw ≤ 0` (one-way; back only on contact),
      launch vz = vx·grade_body, slopes zeroed (free body). Flight: vz under
      −g, `_contact` gates D / rolling resistance / slip-relaxation targets
      to zero (Mz = 0 falls out → yaw rate conserved bit-exact), wheels
      unpinned (driven free-rev, faint bearing drag), pitch eases to the
      flight arc at `air_pitch_rate`. Landing: re-projects the surface from
      raw road values kept fresh during flight, snaps z/vz/pitch/roll,
      absorbs impact as a decaying load spike over `landing_tau` clamped at
      `max_landing_load`, front/rear split by landing attitude; grip loss
      emergent via load-sensitive mu + kappa spike. New CarSpec knobs:
      landing_tau 0.12 s / air_pitch_rate 1.2 / max_landing_load 4.0.
      Gate 3: validate_jumps.py all green — takeoff threshold ±2% around
      analytic launch speed, ramp vz within 2.3%, range 1.2% off the
      parabola, flight energy audit 0.10%, free-rev 3080→6982 rpm,
      landing_g within 2%, loads settle 1.001× static, spin frozen
      bit-exact 59 steps; baseline byte-identical + airborne canary 0;
      Gates 1–2 still green)
- [x] Stage 4 — Hills ON + `--flat`/`--hill-scale` + 3D viewer (2026-06-12:
      defaults flipped (make_track/stadium/_handcrafted hills=True);
      `configure_hills()` = process-wide `--flat`/`--hill-scale` levers in
      run.py, ridge included; curriculum stadiums get seeded hills.
      set_road wired at ALL SIX call sites (app.py, agent.py, ppo_env
      step+reset+reset_at, aiviz ×2, viewer session). session.py: fake
      visual_elevation DELETED; track_payload z = trk.z + bank array +
      elevGain; pose = CAR z/pitch/roll + airborne/airTime/landingG.
      v2 client: server-authoritative attitude (rotation order YZX, roll
      wired), grounded cars drape on the road mesh / airborne cars fly on
      sim z, contact shadow detached to world + shrinks with air height,
      AIR pill w/ airtime + landing-g flash, ridge in the track picker
      (road.js bank ribbon deferred — bank ≡ 0 in hills-v1, payload contract
      shipped). Gate 4: baseline byte-identical (hilly named tracks, no
      set_road = flat behavior — the caller contract holds); all validators
      green; A/B akina scripted lap: hills change pace, flat = 0 air steps;
      ridge end-to-end pure-pursuit send: track→frame→set_road→takeoff→
      flight→landing at 123 km/h over the 28.7 m/s crest, cruise stays
      planted; env race/drift/hybrid 200 hilly steps finite + GA CarAgent
      runs; viewer verified live — ridge renders real elevation (z −5.1…
      +5.5 m over the wire), zero console errors, pose contract confirmed)
- [x] Stage 5 — Obs 58/60 + trainer plumbing + loud checkpoint guards
      (2026-06-12: SensorSpec gains hill_block + refs; sensors append the
      18-dim hill/air block (grade/bank/pitch/vz/height/airborne + 6
      lookahead-grade + 6 lookahead-vcurv at the curvature distances) →
      obs_size 58, obs_dim 60; Observation carries hill_labels +
      lookahead_grade/vcurv raw for viz; first-40 indices verified
      bit-identical with the block on/off. ppo_env: drift/hybrid style +
      drift speed bootstrap gated `not airborne`, drift_frac counts only
      grounded slides; episode info gains airtime/jumps/max_landing_g; ppo
      [eval] line appends `jumps=N air=X.Xs` when nonzero. Checkpoints:
      save dict tagged obs_layout=hills-v1; load_state + load_policy +
      new GA.check_champion (wired at all 3 GA load sites) refuse pre-hills
      files with the loud retrain message — verified against the real
      ppo_race.pt (obs 42) and ga_champion.npz (obs 40). Dashboard: look-
      ahead-grade strip + grade/pitch/vz line + AIR flag; NN INPUT panel
      auto-renders all 58. Gate 5: --ppo 2 / --drift 2 / --train 2 all
      train on hilly curriculum (obs 60, finite weights/normalizer,
      hills-v1 tag, smoke champion obs 58 passes guard); baseline still
      byte-identical (first-40 lock held through the obs growth); Gates
      1–3 green. NOTE: smoke run briefly overwrote ppo_race.pt — restored
      from git; museum intact)
- [x] Stage 6 — Command Center 3D training environment (2026-06-12:
      `_hill_flags()` appends `--flat`/`--hill-scale X` to EVERY action
      (train/continue/live/watch/drive); every train card gains a Terrain
      seg (⛰ Hills / ▭ Flat) + hill-scale slider 0.2–1.5; checkpoint
      metadata carries obs + layout (hills-v1 vs pre-hills, derived from
      the live sensor layout, npz included); checkpoint browser badges
      both kinds w/ tooltips and the →race/→drift activate buttons confirm
      before activating a pre-hills file; watch dropdown + resume pickers
      mark ⚠pre-hills; SSE parser lifts `jumps=N air=X.Xs` off [eval]
      lines and the dock chart gains airtime ✈ / jumps ✈ series; ridge in
      TRACK_NAMES + "touge · JUMPS ⤴" + thumbnail renders. Gate 6 (all
      from the running UI/API): hilly PPO launched w/ --hill-scale 0.6 →
      8 metric points streamed incl. eval_lap → stopped cleanly →
      checkpoint badged hills-v1 obs 60; watching a pre-hills checkpoint
      printed the readable ValueError + 'finished (exit 1)' in-panel (no
      dead process); all 24 museum checkpoints auto-marked ⚠pre-hills;
      zero console errors. Activation defaults still point at the old
      slots — they move to *_hills once Stage 7 retrains)
- [~] Stage 7 — Retraining program — CODE SIDE DONE (2026-06-12): curriculum
      hill ramp live (race `0.4+0.6·d`, drift `0.25+0.5·d` — verified: mean
      elev_gain rises 0.8→7.3 m / 0.9→11.0 m across difficulty, seeds stay
      deterministic); `tools/eval_checkpoint.py` = the Gate 7 measuring stick
      (deterministic act_mean over the named set incl. `ridge` as the
      jump/commitment probe, `--flat` secondary pass + hills-vs-flat delta
      table, jumps/air/landing-g per track, loud pre-hills refusal).
      REMAINING (owner compute): the runs — race generalist (`--ppo`, name it
      `generalist_hills.pt`) → drift generalist → specialists (akina
      hill-drift, ridge send-it) → GA champion refresh; then point the
      Command Center activation defaults at the `*_hills` checkpoints and
      check Gate 7 with eval_checkpoint.py
- [x] Stage 8 — Dashboard readouts, docs, plan/memory closeout (2026-06-12:
      2D dashboard gains the ROAD telemetry line — live grade%/pitch/roll +
      AIR tag w/ airtime + landing-g flash (severity-colored), the 2D twin of
      the 3D HUD pill; README: status rows (hills+jumps ✅ / retrain ⏳),
      ridge + --flat/--hill-scale documented, 18-dim hill/air block in "what
      the agent sees" (58/60), physics-model bullets gain the 2.5D road plane
      + emergent-jump model + validation numbers; HANDOFF: §0.05 hills
      handoff section (contract, gates, levers, retraining notes) + TL;DR
      next-step superseded; VIEWER3D_V2_PLAN Phase 8 status updated; memory
      updated. Banked-corner generation / landing bounce / suspension-travel
      animation stay deferred as the optional extras. NOTE: done out of
      order — Stage 7 (retraining, owner compute) remains; its curriculum
      hill_scale ramp wiring lands with it)
