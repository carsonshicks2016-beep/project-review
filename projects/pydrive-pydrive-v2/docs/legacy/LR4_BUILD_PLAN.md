# Build Plan & Handoff — Add 2012 Land Rover LR4 HSE + Full Driveline Sim

> **Audience:** an AI agent (or human) who has **never seen this project**. This
> document is self-contained: it explains the project, the exact goal, every file
> you must touch, the exact code to write, and how to validate each step. If you
> run out of tokens mid-stage, another agent can resume from the **Status Tracker**
> below and follow the remaining stages verbatim.

---

## STATUS TRACKER  (update this as you go)

| Stage | What | State |
|---|---|---|
| 0 | Spec lock-in (numbers + new `CarSpec` fields agreed) | ✅ done (numbers in §6/§7 below) |
| 1 | Driveline refactor: layout + center/axle diffs + torque converter, **RWD kept bit-identical** | ✅ DONE — see §3.5 |
| 2 | Skyline GT-R → real AWD | ☐ NOT DONE (deferred — independent of LR4; do when ready, §5) |
| 3 | LR4 physics spec (NA V8, ZF6 + TC, AWD locking diffs) | ✅ DONE — 0-60 6.9s, 0-100 7.3s, ~212 km/h, 0.65g SUV handling. Final tune: `tc_capacity=8e-3`, `drag_area=1.35` |
| 4 | LR4 sound (NA cross-plane V8, no turbo/BOV) | ✅ DONE — V8 4th-order branch in `sound.py`; NA handled by pinning `boost=0` for `boost_floor>=1` in `physics._update_boost` (auto-silences turbo+BOV, BOOST gauge reads 0) |
| 5 | LR4 model (tall tan SUV + roof rack) | ✅ DONE — `lr4` mesh + roof rack (`_rack_box`) in `carart.py`, `CAR_COLORS["lr4"]`, `body_roll_gain=1.8` wired into draw_car tilt |
| 6 | Wire-up (`--car lr4`, Command Center) + drive test | ✅ DONE — `run.py` choices + `app.js CARS`; drives end-to-end on akina |

**LR4 IS COMPLETE.** Remaining optional polish: (a) LR4 headlight/taillight decals in `draw_car` (it currently has none — falls through the per-car decal blocks); (b) dashboard BOOST gauge could show "N/A" for NA cars (`dashboard.py`); (c) train an LR4 race specialist (`python3 run.py --ppo 800 --car lr4`) — existing checkpoints drive it but understeer/plow since untrained on a heavy AWD SUV. Stage 2 (Skyline AWD) is still available, independent of the LR4.

**Decisions already made by the project owner:**
- Car: **2012 Land Rover LR4 HSE**, **tan** body, with a **roof rack**.
- Driveline depth: **FULL** — torque converter + open/locking **center and rear diffs**.
- **Upgrade the Skyline GT-R to real AWD** as part of this (it is currently faked as RWD).

---

## 0. COLD-START ORIENTATION (read this first)

### What this project is
`Supra Drift` — a from-scratch 2D top-down neuro-evolution + reinforcement-learning
**drifting/racing simulator** in pure Python (numpy + pygame + torch + flask). It
has real four-wheel vehicle physics, procedural tracks, a genetic-algorithm path
and a PyTorch PPO path, live synthesized engine audio, and a Flask "Command
Center" web dashboard. Project root: `~/Desktop/Supra Ai 2/`.

### Conventions / environment
- **macOS**, use `python3` / `pip3` (NOT `python`/`pip`). Torch is CPU-only.
- Run a headless test by setting `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`
  before `python3` (lets pygame init with no display/audio device).
- There is NO git repo here (do not rely on `git`).
- After ANY change: `python3 -c "import compileall,sys; sys.exit(0 if compileall.compile_dir('supra', quiet=1) else 1)"`.

### Architecture map (the files that matter)
| File | Role |
|---|---|
| `supra/config.py` | **`CarSpec`** dataclass (~50 params = a car), car preset functions (`supra()`, `rx7()`, `skyline()`), `PRESETS` dict, `get_car()`. Also `SimSpec` (dt, fps). |
| `supra/physics.py` | **`Vehicle`** four-wheel dynamics. Tyre model (`_tyre_forces`, `_pacejka`), load transfer (`_update_loads`), engine (`_engine_torque`, `_update_boost`), **`_drivetrain`** (the thing Stage 1 rewrites), `step()`. Wheel order is ALWAYS `[FL, FR, RL, RR]` = indices `0,1,2,3`; `FRONT=[0,1]`, `REAR=[2,3]`. |
| `supra/sound.py` | `EngineAudio` — real-time additive synth in a `sounddevice` callback. Per-car branches keyed on `self.car_name` ("supra"/"rx7"/"skyline"). `BLOCK=2048`. |
| `supra/carart.py` | Procedural low-poly faux-3D car renderer. `CAR_COLORS` dict, `get_car_mesh(name, HL, HW)` returns per-car (vertices, faces), `add_wheels_to_mesh`, `draw_car(...)` with per-car decals. |
| `supra/app.py` | Keyboard/agent drive view (`run()`), `Camera`, `AutoBox` (automatic gearbox: returns clutch + shift signals), `draw_road()`. |
| `supra/app_ppo.py`, `supra/app_ga.py` | Live training views. |
| `supra/ppo_env.py` | `SupraEnv` — RL environment; maps agent action `[steer, long(+thr/-brk), (handbrake)]` → `Controls`; uses `AutoBox` for clutch/shift. |
| `run.py` | CLI entry. `--car {supra,rx7,skyline}` choices live here (~line 28). Dispatches `--drive`, `--watch*`, training. |
| `command-center/server.py` | Flask dashboard; spawns `run.py`. `--car` is passed via UI params; thumbnails rendered headlessly in `_render_thumb`. |

### How a car flows through the sim
`get_car("supra")` → `CarSpec` → `Vehicle(spec, SimSpec())`. Each tick:
`Vehicle.step(Controls(...))` calls `_update_loads` (vertical tyre loads incl.
weight transfer + downforce + suspension lag) → `_drivetrain` (engine→coupling→
gearbox→diff→wheels, sub-stepped, returns tyre Fx,Fy) → integrates body motion.
`AutoBox.update(veh, throttle, dt)` produces the clutch + shift commands so the
agent only needs steer/throttle/brake.

---

## 1. TRAINING-SAFETY INVARIANTS (do not violate)

The RL policies (GA `.npz`, PPO `.pt` checkpoints in the repo root) eat a
**fixed-dimension observation** and output `[steer, throttle, brake (, handbrake)]`.
To keep every existing checkpoint valid:

1. **NEVER change the observation vector dimension or meaning** (`supra/sensors.py`,
   `ppo_env.py` obs assembly). Adding cars must NOT touch obs. (The LR4 needs no
   obs change — it is just new spec numbers.)
2. **Keep RWD + clutch numerically identical.** Existing cars (`supra`, `rx7`) are
   RWD + clutch. Stage 1 must preserve their trajectories bit-for-bit. The
   regression test in §4.4 is the gate — it MUST pass before Stage 2.
3. **Adding new `CarSpec` fields is safe** as long as they have defaults that make
   existing cars behave exactly as before (`drive_layout="rwd"`, `coupling="clutch"`,
   center split contributing 0, new diff coefs not used on the RWD path).
4. **The Skyline change (Stage 2) is the one intentional exception** — its dynamics
   change, so its specialist checkpoints become slightly mistuned (still load/run;
   obs unchanged). An optional Skyline retrain is fine; generalist policies are
   robust. Do NOT also change supra/rx7.

---

## 2. CAR REGISTRATION CHECKLIST (every place a new car name must appear)

When adding `"lr4"`, it must be registered in ALL of these:
1. `supra/config.py`: a `def lr4() -> CarSpec:` preset + add to `PRESETS` dict.
2. `supra/carart.py`: `CAR_COLORS["lr4"] = (...)`; a branch in `get_car_mesh`; (optional) per-car decals in `draw_car`.
3. `supra/sound.py`: car branch(es) keyed on `self.car_name == "lr4"` (engine, and the NA "no turbo/BOV" handling).
4. `run.py`: add `"lr4"` to `--car choices=[...]` (~line 28).
5. **Command Center cars now auto-derive from `PRESETS` — no hand-edit needed.**
   `server.py` does `from supra.config import PRESETS; CARS = list(PRESETS.keys())`
   and exposes `GET /api/cars`; the frontend (`app.js`) fetches `/api/cars` to build
   the chassis selectors (with a hardcoded fallback). So a new preset shows up in
   the dashboard automatically. ⚠️ **A `server.py` change needs a dashboard
   RESTART** (stop + re-run `python3 command-center/server.py`) and a browser
   refresh. (Historical bug: the server used to keep its OWN stale `CARS` list and
   silently fell back to "supra" for any car not in it — that's why a freshly added
   car looked/sounded like the Supra in the dashboard.)
6. (Optional) AI: existing checkpoints will *drive* it (obs-compatible) but plow/understeer; a per-car specialist retrain is optional and touches no other checkpoint.

---

## 3. STAGE 1 — DRIVELINE REFACTOR  (do this now)

**Goal:** generalize `Vehicle._drivetrain` from hardcoded RWD+clutch to support
drive layout (rwd/fwd/awd), a center diff (open/lsd/locked) with a torque split,
per-axle diffs (open/lsd/locked), and a torque-converter coupling — **while
keeping the existing RWD+clutch path bit-identical.**

### 3.1 Add fields to `CarSpec` (`supra/config.py`)
Insert into the `@dataclass class CarSpec` (after the existing `# --- limited-slip
differential ---` block, around the `lsd_coef` line). Note: the existing
`lsd_coef` now semantically means the **REAR** axle diff coefficient.

```python
    # --- drivetrain layout ---
    drive_layout: str = "rwd"          # "rwd" | "fwd" | "awd"
    center_split_front: float = 0.0    # AWD: fraction of drive torque to FRONT axle
                                       #   rwd=0.0, fwd=1.0, awd e.g. 0.5 (LR4) / 0.35 (R34)
    center_diff: str = "open"          # "open" | "lsd" | "locked"  (AWD only)
    center_lsd_coef: float = 150.0     # locking torque per rad/s of front-rear shaft delta
    front_lsd_coef: float = 0.0        # front-axle diff locking (0 = open)
    # (existing `lsd_coef` is the REAR-axle diff locking coefficient)

    # --- coupling (engine -> gearbox) ---
    coupling: str = "clutch"           # "clutch" | "torque_converter"
    # torque-converter params (used only when coupling == "torque_converter")
    tc_capacity: float = 3.5e-4        # pump capacity factor: T_pump = tc_capacity*w_imp^2*(1-sr^2)
    tc_mult_max: float = 2.2           # torque multiplication at stall (speed ratio 0)
    tc_coupling_sr: float = 0.88       # speed ratio where torque ratio reaches 1.0
    tc_lockup_sr: float = 0.92         # speed ratio above which the lockup clutch blends in
```

### 3.2 Rewrite `Vehicle._drivetrain` (`supra/physics.py`)
Strategy: keep the **original loop verbatim as a fast path** for
`drive_layout=="rwd" and coupling=="clutch"` (guarantees bit-identical existing
cars), and add a **general path** for everything else. Shared setup stays above
the branch.

Replace the body of `_drivetrain` (currently lines ~330–405) with:

```python
    def _drivetrain(self, c: Controls, v_long, denom, D, dt):
        s = self.spec
        n = self.sim.drivetrain_substeps
        sub = dt / n
        r = s.wheel_radius
        tan_alpha = np.tan(self.alpha_lag)

        brake_axle = np.array([
            c.brake * s.max_brake_torque * s.brake_bias / 2.0,
            c.brake * s.max_brake_torque * s.brake_bias / 2.0,
            c.brake * s.max_brake_torque * (1.0 - s.brake_bias) / 2.0,
            c.brake * s.max_brake_torque * (1.0 - s.brake_bias) / 2.0,
        ])
        brake_axle[REAR] += c.handbrake * s.handbrake_torque / 2.0

        self._update_boost(c.throttle, dt)

        tau_long = s.relaxation_length_long / np.maximum(np.abs(v_long), self.sim.relax_speed_floor)
        a_long = sub / (tau_long + sub)

        # ---- FAST PATH: RWD + clutch (existing cars) — keep bit-identical ----
        if s.drive_layout == "rwd" and s.coupling == "clutch":
            free_roll = np.array([True, True, False, False]) & (brake_axle <= 1e-6)
            roll_w = v_long / r
            Fx = np.zeros(4); Fy = np.zeros(4)
            for _ in range(n):
                self.wheel_w[free_roll] = roll_w[free_roll]
                kappa_inst = np.clip((self.wheel_w * r - v_long) / denom, -4.0, 4.0)
                self.kappa_lag += a_long * (kappa_inst - self.kappa_lag)
                Fx, Fy = self._tyre_forces(self.kappa_lag, tan_alpha, D)
                self.wheel_sr = self.kappa_lag.copy()

                Te = self._engine_torque(c.throttle) - s.engine_friction * self.engine_w
                idle_w = self._rpm_to_w(s.idle_rpm)
                if self.engine_w < idle_w:
                    Te += min(s.idle_torque_max, s.idle_gain * (idle_w - self.engine_w))
                drive_w = 0.5 * (self.wheel_w[RL] + self.wheel_w[RR])
                d_w = self.engine_w - drive_w * self.total_ratio
                Tc = c.clutch * s.clutch_capacity * np.tanh(d_w / s.clutch_slip_ref)
                self.engine_w += sub * (Te - Tc) / s.engine_inertia
                self.engine_w = max(self.engine_w, 0.0)

                T_drive = Tc * self.total_ratio * s.drivetrain_efficiency
                t_lsd = s.lsd_coef * (self.wheel_w[RL] - self.wheel_w[RR])
                T_wheel = np.zeros(4)
                T_wheel[RL] = T_drive / 2.0 - t_lsd
                T_wheel[RR] = T_drive / 2.0 + t_lsd

                self.wheel_w += sub * (T_wheel - Fx * r) / s.wheel_inertia
                dw_brake = sub * brake_axle / s.wheel_inertia
                self.wheel_w = np.where(
                    np.abs(self.wheel_w) <= dw_brake, 0.0,
                    self.wheel_w - np.sign(self.wheel_w) * dw_brake)
            return Fx, Fy

        # ---- GENERAL PATH: fwd / awd and/or torque converter ----
        driven = self._driven_mask()                  # bool[4]
        free_roll = (~driven) & (brake_axle <= 1e-6)
        roll_w = v_long / r
        Fx = np.zeros(4); Fy = np.zeros(4)
        for _ in range(n):
            self.wheel_w[free_roll] = roll_w[free_roll]
            kappa_inst = np.clip((self.wheel_w * r - v_long) / denom, -4.0, 4.0)
            self.kappa_lag += a_long * (kappa_inst - self.kappa_lag)
            Fx, Fy = self._tyre_forces(self.kappa_lag, tan_alpha, D)
            self.wheel_sr = self.kappa_lag.copy()

            # engine torque
            Te = self._engine_torque(c.throttle) - s.engine_friction * self.engine_w
            idle_w = self._rpm_to_w(s.idle_rpm)
            if self.engine_w < idle_w:
                Te += min(s.idle_torque_max, s.idle_gain * (idle_w - self.engine_w))

            # transmission input speed seen at the engine, from the driven wheels
            drive_w = float(np.mean(self.wheel_w[driven]))
            omega_tur = drive_w * self.total_ratio    # turbine / input-shaft speed

            # --- coupling: clutch OR torque converter -> T_in (gearbox input torque) ---
            if s.coupling == "torque_converter":
                w_imp = max(self.engine_w, 1.0)
                sr = float(np.clip(omega_tur / w_imp, 0.0, 1.0))
                T_pump = s.tc_capacity * w_imp * w_imp * max(0.0, 1.0 - sr * sr)
                TR = 1.0 + (s.tc_mult_max - 1.0) * max(0.0, 1.0 - sr / max(s.tc_coupling_sr, 1e-3))
                T_in = T_pump * TR
                # lockup clutch blends toward rigid coupling at high speed ratio
                if sr > s.tc_lockup_sr:
                    lock = (sr - s.tc_lockup_sr) / max(1.0 - s.tc_lockup_sr, 1e-3)
                    lock = float(np.clip(lock, 0.0, 1.0)) * c.clutch
                    d_w = self.engine_w - omega_tur
                    Tc_lock = s.clutch_capacity * np.tanh(d_w / s.clutch_slip_ref)
                    T_pump = (1.0 - lock) * T_pump + lock * Tc_lock
                    T_in = (1.0 - lock) * T_in + lock * Tc_lock
                self.engine_w += sub * (Te - T_pump) / s.engine_inertia
            else:  # clutch
                d_w = self.engine_w - omega_tur
                Tc = c.clutch * s.clutch_capacity * np.tanh(d_w / s.clutch_slip_ref)
                T_in = Tc
                self.engine_w += sub * (Te - Tc) / s.engine_inertia
            self.engine_w = max(self.engine_w, 0.0)

            T_drive = T_in * self.total_ratio * s.drivetrain_efficiency

            # --- center diff: split T_drive front/rear, plus locking transfer ---
            front_axle_T = T_drive * s.center_split_front
            rear_axle_T = T_drive * (1.0 - s.center_split_front)
            if s.drive_layout == "awd" and s.center_diff in ("lsd", "locked"):
                w_front = 0.5 * (self.wheel_w[FL] + self.wheel_w[FR])
                w_rear = 0.5 * (self.wheel_w[RL] + self.wheel_w[RR])
                coef = s.center_lsd_coef * (8.0 if s.center_diff == "locked" else 1.0)
                t_center = coef * (w_front - w_rear)   # +: front faster -> shove torque rearward
                front_axle_T -= t_center
                rear_axle_T += t_center

            # --- per-axle diffs ---
            T_wheel = np.zeros(4)
            if s.center_split_front > 0.0:  # front axle is driven
                t_lsd_f = s.front_lsd_coef * (self.wheel_w[FL] - self.wheel_w[FR])
                T_wheel[FL] = front_axle_T / 2.0 - t_lsd_f
                T_wheel[FR] = front_axle_T / 2.0 + t_lsd_f
            if s.center_split_front < 1.0:  # rear axle is driven
                t_lsd_r = s.lsd_coef * (self.wheel_w[RL] - self.wheel_w[RR])
                T_wheel[RL] = rear_axle_T / 2.0 - t_lsd_r
                T_wheel[RR] = rear_axle_T / 2.0 + t_lsd_r

            self.wheel_w += sub * (T_wheel - Fx * r) / s.wheel_inertia
            dw_brake = sub * brake_axle / s.wheel_inertia
            self.wheel_w = np.where(
                np.abs(self.wheel_w) <= dw_brake, 0.0,
                self.wheel_w - np.sign(self.wheel_w) * dw_brake)
        return Fx, Fy
```

Add this helper method near `_drivetrain`:

```python
    def _driven_mask(self) -> np.ndarray:
        """Bool[4] of which wheels receive drive torque, per layout."""
        layout = self.spec.drive_layout
        if layout == "awd":
            return np.array([True, True, True, True])
        if layout == "fwd":
            return np.array([True, True, False, False])
        return np.array([False, False, True, True])   # rwd
```

**FWD note:** the general path drives the front wheels but applies drive torque in
the *body* x via the same `T_wheel - Fx*r` path; the front patches are already
rotated by steer in `step()`. Torque-steer emerges naturally. No special case
needed for FWD beyond `center_split_front=1.0` + `drive_layout="fwd"`.

### 3.3 Telemetry (optional, recommended)
`Vehicle.telemetry()` returns `boost`, etc. Leave as-is for Stage 1. (For NA cars
in Stage 3, `boost` will read ~0; the dashboard BOOST gauge handling is a cosmetic
Stage 4/6 polish — see §7.)

### 3.4 REGRESSION TEST — the Stage 1 gate (MUST pass before Stage 2)
Create `tools/regression_drivetrain.py` (or run inline). It drives `supra` and
`rx7` through a fixed control script for N steps and prints a checksum of the
trajectory. Run it **before** editing (capture baseline) and **after** (confirm
identical). Because the RWD+clutch fast path is the original code verbatim, the
checksums MUST match exactly.

```python
# SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 tools/regression_drivetrain.py
import numpy as np
from supra.config import get_car, SimSpec
from supra.physics import Vehicle, Controls

def run_car(name):
    v = Vehicle(get_car(name), SimSpec()); v.reset(0,0,0,speed=0.0)
    rng = np.random.default_rng(0); rec=[]
    for i in range(3000):
        thr = 0.8 if i % 600 < 400 else 0.0
        brk = 0.6 if i % 600 >= 500 else 0.0
        steer = 0.5*np.sin(i*0.01)
        v.step(Controls(steer=steer, throttle=thr, brake=brk,
                        clutch=1.0 if thr>0 else 0.0))
        rec.append((v.x, v.y, v.yaw, v.vx, v.vy, v.r, v.engine_w, *v.wheel_w))
    a = np.array(rec)
    return float(np.sum(a)), a

for name in ("supra","rx7"):
    chk,_ = run_car(name)
    print(f"{name}: checksum={chk:.10e}")
```

**Run it with** `PYTHONPATH="$PWD"` (the script lives in `tools/`):
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONPATH="$PWD" python3 tools/regression_drivetrain.py
```

**Procedure:** (1) BEFORE any edit, run it, save the two checksums. (2) Make the
§3.1+§3.2 edits. (3) Run again. (4) The two numbers per car MUST be **identical**.
If they differ, the RWD fast path was altered — diff it against the original and
fix. Only proceed to Stage 2 when identical.

### 3.5 STAGE 1 RESULT (completed)
- `CarSpec` fields added (§3.1) incl. `body_roll_gain` (for §8). `lsd_coef` is now
  documented as the REAR diff coefficient.
- `_drivetrain` rewritten (§3.2): verbatim RWD+clutch fast path + general
  FWD/AWD/torque-converter path; `_driven_mask()` helper added.
- **Regression gate PASSED — byte-identical.** Baseline = after:
  `supra: 2.163028270861e+06`, `rx7: 2.154679517578e+06`,
  `skyline: 2.265068194448e+06` (skyline still RWD until Stage 2).
- General path functionally verified: AWD-locked equalizes all 4 wheels, FWD spins
  the fronts, RWD spins the rears, torque-converter launches cleanly — all finite.
- Existing checkpoints are SAFE (RWD numerics unchanged, obs untouched).

A full smoke check that all three cars still run + the engine doesn't NaN:
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -c "
from supra.config import get_car,SimSpec; from supra.physics import Vehicle,Controls
import numpy as np
for n in ('supra','rx7','skyline'):
    v=Vehicle(get_car(n),SimSpec()); v.reset()
    for i in range(1200): v.step(Controls(steer=0.3,throttle=1.0,clutch=1.0))
    assert np.isfinite(v.x) and v.speed<200, n
    print(n,'ok  v=%.1f km/h rpm=%.0f'%(v.speed*3.6,v.rpm))
"
```

---

## 5. STAGE 2 — Skyline GT-R → real AWD

Edit `def skyline()` in `supra/config.py`. Add to its `CarSpec(...)` call:
```python
        drive_layout="awd",
        center_split_front=0.35,     # ATTESA-style rear bias (most torque rear)
        center_diff="lsd",
        center_lsd_coef=180.0,
        # lsd_coef (rear) stays as-is; front_lsd_coef left 0 (open front)
```
**Validate:** drive it (`python3 run.py --drive --car skyline`) — it should pull
straight on corner exit and resist power-oversteer more than before, but still be
tail-adjustable. Telemetry sanity (headless):
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -c "
from supra.config import get_car,SimSpec; from supra.physics import Vehicle,Controls
v=Vehicle(get_car('skyline'),SimSpec()); v.reset()
import time
for i in range(900): v.step(Controls(throttle=1.0,clutch=1.0))
print('0-100ish: %.1f km/h after 7.5s'%(v.speed*3.6))
"
```
Existing Skyline checkpoints still load (obs unchanged) but are now mistuned —
offer the owner an optional `--drift`/`--ppo` Skyline specialist retrain. Do NOT
touch supra/rx7.

---

## 6. STAGE 3 — LR4 physics spec

Add to `supra/config.py` (after `def skyline()`), then register in `PRESETS`.

```python
def lr4() -> CarSpec:
    """2012 Land Rover LR4 HSE — tall, heavy, NA 5.0 V8, full-time AWD, soft air
    suspension. The understeery anti-sports-car: big roll, lazy yaw, planted."""
    return CarSpec(
        name="lr4",
        # chassis / mass (tall + heavy SUV)
        mass=2650.0,
        wheelbase=2.885,
        track_width=1.605,
        cg_height=0.86,            # ~2x a sports car -> huge load transfer + roll
        front_weight=0.51,
        yaw_inertia=4800.0,        # long + heavy -> lazy rotation
        # tyres (tall all-season, lower grip)
        mu=1.00,
        wheel_radius=0.38,         # 255/55 R19
        wheel_inertia=2.4,         # big heavy wheels
        load_sensitivity=6.0e-5,
        rolling_resistance=0.016,
        # handling balance: understeer bias + slow soft responses
        roll_front_frac=0.58,      # >0.5 -> understeer
        steer_angle_max_deg=34.0,
        steer_rate_deg_s=220.0,    # slow rack
        offtrack_grip=0.85,        # it's actually good off-tarmac
        # aero: a brick with a roof rack, no downforce
        drag_area=1.20,
        side_drag_area=3.2,
        downforce_ClA=0.05,
        aero_balance=0.5,
        # powertrain: NA 5.0 V8 (375 hp / ~508 Nm @ 3500), 6500 redline
        engine_inertia=0.42,
        idle_rpm=650.0,
        redline_rpm=6500.0,
        cutoff_rpm=6700.0,
        torque_curve=[
            (650, 300), (1500, 400), (2500, 470), (3500, 508),
            (4500, 495), (5500, 460), (6500, 400), (6700, 0),
        ],
        # ZF 6HP28 6-speed automatic + final drive
        gear_ratios=[4.17, 2.34, 1.52, 1.14, 0.87, 0.69],
        final_drive=3.54,
        drivetrain_efficiency=0.85,   # AWD + auto losses
        # NA: no boost dependence
        boost_floor=1.0,
        spool_up_tau=0.30, spool_down_tau=0.20,   # unused at boost_floor=1 but keep sane
        # FULL driveline: torque converter + AWD locking center + rear diff
        coupling="torque_converter",
        tc_capacity=4.0e-4, tc_mult_max=2.2, tc_coupling_sr=0.88, tc_lockup_sr=0.92,
        drive_layout="awd",
        center_split_front=0.5,       # symmetric full-time 4WD
        center_diff="locked",         # locking center diff
        center_lsd_coef=200.0,
        lsd_coef=140.0,               # rear diff
        front_lsd_coef=60.0,          # mild front
        # brakes: big but soft pedal, lots of dive
        max_brake_torque=3200.0,
        brake_bias=0.60,
        handbrake_torque=2600.0,
        # soft air suspension -> slow load settling = visible roll/dive
        suspension_tau=0.26,
    )
```
Then: `PRESETS = {"supra": supra, "rx7": rx7, "skyline": skyline, "lr4": lr4}`.

**Validate targets** (tune torque_curve / mu / drag toward these):
- 0–100 km/h ≈ **7.5–8.0 s**; top speed ≈ **190–200 km/h** (drag-limited).
- Strong understeer; slow turn-in; pronounced lateral load transfer.
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -c "
from supra.config import get_car,SimSpec; from supra.physics import Vehicle,Controls
v=Vehicle(get_car('lr4'),SimSpec()); v.reset(); t=0
while v.speed*3.6<100 and t<15: v.step(Controls(throttle=1.0,clutch=1.0)); t+=1/120
print('0-100 in %.1fs'%t)
for i in range(4000): v.step(Controls(throttle=1.0,clutch=1.0))
print('top ~ %.0f km/h, rpm %.0f gear %d'%(v.speed*3.6,v.rpm,v.gear))
"
```

---

## 7. STAGE 4 — LR4 sound (NA cross-plane V8)

Edit `supra/sound.py`. The synth currently ALWAYS adds turbo whine + BOV. For the
LR4 (NA), add an `lr4` branch and suppress turbo/BOV:
- **Engine harmonics** (`_callback`, the `if car == "rx7" / elif skyline / else`
  block ~lines 207–221): add `elif car == "lr4":` with a deep V8 burble — cross-plane
  V8 dominant orders ~ `[0.5, 1.0, 2.0, 4.0, 6.0, 8.0]` (the 0.5 gives the lopey
  half-order rumble), amps emphasizing low orders, modest `drive` (1.4 + 3.0*thr).
- **Turbo** (~lines 237–264): wrap so `if car == "lr4": turbo = np.zeros(n)` (NA).
- **BOV** (the `if self.bov_env > 1e-3:` block ~266): add `if car == "lr4": pass`
  / skip — NA has no blow-off. (The `update()` BOV-arming can stay; it just never
  fires audibly. Optionally gate arming when `car=="lr4"`.)
- **Backfire/overrun** (~166–171): a soft, infrequent V8 burble pop is fine; leave
  default or give `lr4` a low `prob` and bassier `filter_size=10`.
- Add an entry so `update()` knows the redline (already generic via `veh.spec.redline_rpm`).

**Cosmetic:** the dashboard BOOST gauge will read ~0 for NA. Optional: in
`supra/dashboard.py` show "N/A" or hide the BOOST bar when `spec.boost_floor>=1.0`.

Validate: `python3 run.py --drive --car lr4` and listen, or headless-benchmark the
callback as in earlier audio tests (must stay <~5 ms/buffer, finite output).

---

## 8. STAGE 5 — LR4 model (tall tan SUV + roof rack)

Edit `supra/carart.py`:
1. `CAR_COLORS["lr4"] = (190, 176, 142)`  # tan / Nara-bronze sand.
2. In `get_car_mesh(name, HL, HW)`, add an `elif name == "lr4":` block. Author a
   **tall, boxy, upright** mesh: nearly vertical windshield/backlight, long flat
   roof, high beltline, big greenhouse. Reuse the same vertex index convention
   (centerline 0–7, left 8–19 mirrored to 20–31). Make the roof z-height clearly
   taller than the sports cars (their roof ~1.2; use ~1.55–1.7) so it towers in the
   2.5D view. Then append **roof-rack** geometry: 2 longitudinal rails + ~3 cross
   bars as thin boxes on top (extra vertices/faces, color a dark `(40,42,46)`).
3. (Optional) In `draw_car`, add an `elif name == "lr4":` decal block: big square
   headlights/taillights, black wheel-arch cladding, a front grille rectangle.
4. The renderer tilts the body by lateral g already (`tilt_roll`, ~line 326).
   For the SUV "lean", scale roll per car: multiply `tilt_roll`/`tilt_pitch` by a
   `getattr(spec, "body_roll_gain", 1.0)` and add `body_roll_gain: float = 1.0` to
   CarSpec (LR4 ≈ 1.8). (This is a small CarSpec + carart addition — keep default
   1.0 so other cars are unchanged.)

Render check (headless), pattern from existing tests:
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -c "
import pygame; pygame.init()
from supra.config import get_car,SimSpec; from supra.physics import Vehicle
from supra.carart import draw_car
s=pygame.Surface((600,400)); spec=get_car('lr4'); v=Vehicle(spec,SimSpec()); v.reset(0,0,0)
draw_car(s, lambda x,y:(int(300+x*18),int(200+y*18)), 18, v, spec)
pygame.image.save(s,'/tmp/lr4.png'); print('saved /tmp/lr4.png')
"
```

---

## 9. STAGE 6 — wire-up + drive test

1. `run.py` ~line 28: `--car choices=["supra","rx7","skyline","lr4"]`.
2. Command Center: in `command-center/static/` find the car selector (grep the
   JS/HTML for `skyline`) and add an `lr4` option labelled e.g. "Land Rover LR4".
   `server.py` passes `--car` straight through; no server logic change needed.
3. Thumbnail auto-generates via `_render_thumb` (it calls `get_car`/`draw_car`).
   Delete any stale `command-center/thumbs/lr4.png` to force a re-render.
4. Final checks:
   - `python3 -c "import compileall,sys; sys.exit(0 if compileall.compile_dir('supra',quiet=1) else 1)"`
   - `python3 -m py_compile run.py`
   - `python3 run.py --drive --car lr4` (drive it; confirm feel + sound + model).
5. (Optional) AI: `python3 run.py --ppo 800 --car lr4` to train an LR4 race
   specialist; doesn't affect other checkpoints.

---

## 10. VALIDATION CHEATSHEET
```bash
# compile everything
python3 -c "import compileall,sys; sys.exit(0 if compileall.compile_dir('supra',quiet=1) else 1)"
# Stage-1 regression gate (supra/rx7 must be byte-identical before vs after)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYTHONPATH="$PWD" python3 tools/regression_drivetrain.py
# all three cars still run, no NaN
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python3 -c "from supra.config import get_car,SimSpec; from supra.physics import Vehicle,Controls; import numpy as np; [(\
 lambda v:[v.step(Controls(steer=0.3,throttle=1.0,clutch=1.0)) for _ in range(1200)] and print(n,'ok'))(Vehicle(get_car(n),SimSpec()).reset()) for n in ('supra','rx7','skyline')]"
```

## 11. GOTCHAS / ROLLBACK
- **Float identity:** `x*1.0 == x` and `x/2.0 == x*0.5` exactly in IEEE — the RWD
  fast path is preserved precisely by keeping it as a separate verbatim branch.
- **`Controls.clutch` default is 1.0**; AutoBox drives it during play. The TC path
  uses `c.clutch` only for the lockup blend.
- **Torque converter stability:** if the LR4 creeps at idle or oscillates, lower
  `tc_capacity` or raise `tc_coupling_sr`; if it won't launch, raise `tc_capacity`.
- **Do not change `SimSpec.dt`** (1/120) — it would alter every car's dynamics and
  all checkpoints' effective behavior.
- **If the regression test fails:** revert `_drivetrain` to the original (the
  fast-path block IS the original) and re-diff; the general path must not be on the
  RWD+clutch code path.
```
