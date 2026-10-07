"""Pre-hills physics baseline — Stage 0 of PHYSICS_3D_PLAN.md.

Records byte-exact checksums of the simulator BEFORE any elevation work, then
verifies them after every subsequent stage. Three locks:

  1. drivetrain   — the regression_drivetrain.py run, verbatim, for
                    supra (RWD clutch fast path), rx7 (same path, different
                    spec) and skyline (ATTESA AWD general path).
  2. track runs   — supra driven by a deterministic scripted controller on
                    named_track("club") and ("akina"): per step
                    frame -> surface_grip -> Controls, exactly the agent.py
                    call pattern. Locks the physics<->track coupling
                    (including off-track grip transitions on akina).
  3. sensors      — per-dimension sums of SensorSuite.observe().vector[:40]
                    along those runs, plus the full first observation.
                    Only the FIRST 40 dims are locked, so this keeps passing
                    after Stage 5 appends the hill block — but fails on any
                    reorder / renorm of the existing dims.

Every compared value is stored as a float hex string (byte-exact round-trip).
`airborne_steps` is counted via getattr() so the Stage 3 canary ("a flat
track can never take off") works the moment the attribute exists; until then
it is trivially 0.

This harness deliberately never calls Vehicle.set_road(), so it stays valid
after Stage 4 flips hills on: a caller that opts out of the road data must
get exact pre-hills behavior forever (that IS the contract being tested).

Usage (house convention: run from the repo root):
  PYTHONPATH="$PWD" python3 tools/regression_baseline.py --write   # once, pre-edit
  PYTHONPATH="$PWD" python3 tools/regression_baseline.py           # gate: compare
Exits non-zero on any mismatch.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supra.config import SensorSpec, SimSpec, get_car
from supra.physics import Controls, Vehicle
from supra.sensors import SensorSuite
from supra.track import named_track

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "baselines_prehills.json")
OBS_LOCK = 40          # the pre-hills obs dims this harness locks
STEPS = 3000
OBS_EVERY = 10


def hx(v) -> str:
    return float(v).hex()


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# 1. drivetrain runs (verbatim from tools/regression_drivetrain.py)
# --------------------------------------------------------------------------- #
def run_drivetrain(name: str) -> np.ndarray:
    v = Vehicle(get_car(name), SimSpec()); v.reset(0, 0, 0, speed=0.0)
    rec = []
    for i in range(STEPS):
        thr = 0.8 if i % 600 < 400 else 0.0
        brk = 0.6 if i % 600 >= 500 else 0.0
        steer = 0.5 * np.sin(i * 0.01)
        v.step(Controls(steer=steer, throttle=thr, brake=brk,
                        clutch=1.0 if thr > 0 else 0.0))
        rec.append((v.x, v.y, v.yaw, v.vx, v.vy, v.r, v.engine_w, *v.wheel_w))
    return np.array(rec)


# --------------------------------------------------------------------------- #
# 2+3. track-coupled scripted run + sensor sums
# --------------------------------------------------------------------------- #
def run_track(track_name: str) -> dict:
    trk = named_track(track_name)
    veh = Vehicle(get_car("supra"), SimSpec())
    sensors = SensorSuite(SensorSpec())
    sx, sy, syaw = trk.start_pose()
    veh.reset(sx, sy, syaw)

    rec = []
    obs_acc = np.zeros(OBS_LOCK, dtype=np.float64)
    obs_first = None
    obs_count = 0
    offtrack_steps = 0
    airborne_steps = 0

    for i in range(STEPS):
        fr = trk.frame(veh.x, veh.y)
        veh.surface_grip = veh.spec.offtrack_grip if fr["off_track"] else 1.0
        offtrack_steps += int(fr["off_track"])
        airborne_steps += int(bool(getattr(veh, "airborne", False)))

        if i % OBS_EVERY == 0:
            vec = sensors.observe(veh, trk).vector
            if obs_first is None:
                obs_first = vec[:OBS_LOCK].astype(np.float64).copy()
            obs_acc += vec[:OBS_LOCK].astype(np.float64)
            obs_count += 1

        # deterministic chase-the-centerline controller: P on heading error +
        # lateral offset, a crude speed governor, rpm-window autoshift. It is
        # NOT meant to drive well — akina's hairpins push it off-track, which
        # exercises the surface_grip transitions on purpose.
        herr = _wrap(fr["heading"] - veh.yaw)
        steer = float(np.clip(1.4 * herr - 0.045 * fr["lateral"], -1.0, 1.0))
        spd = veh.speed
        thr = 0.55 if spd < 22.0 else 0.0
        brk = 0.5 if spd > 26.0 else 0.0
        up = veh.rpm > 5800.0 and veh.gear < len(veh.spec.gear_ratios)
        down = veh.rpm < 2200.0 and veh.gear > 1
        veh.step(Controls(steer=steer, throttle=thr, brake=brk,
                          clutch=1.0 if thr > 0 else 0.0,
                          shift_up=up, shift_down=down))
        rec.append((veh.x, veh.y, veh.yaw, veh.vx, veh.vy, veh.r,
                    veh.engine_w, *veh.wheel_w))

    traj = np.array(rec)
    return {
        "traj_checksum": hx(np.sum(traj)),
        "final_state": [hx(v) for v in traj[-1]],
        "obs_dim_sums": [hx(v) for v in obs_acc],
        "obs_first": [hx(v) for v in obs_first],
        "obs_count": obs_count,
        "offtrack_steps": offtrack_steps,
        "airborne_steps": airborne_steps,
    }


def compute() -> dict:
    out = {"drivetrain": {}, "tracks": {}}
    for name in ("supra", "rx7", "skyline"):
        a = run_drivetrain(name)
        out["drivetrain"][name] = {
            "checksum": hx(np.sum(a)),
            "final_state": [hx(v) for v in a[-1]],
        }
    for tname in ("club", "akina"):
        out["tracks"][tname] = run_track(tname)
    return out


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def diff(expected, got, path="") -> list:
    bad = []
    if isinstance(expected, dict):
        for k in expected:
            if k not in got:
                bad.append(f"{path}/{k}: MISSING in current run")
            else:
                bad += diff(expected[k], got[k], f"{path}/{k}")
        return bad
    if isinstance(expected, list):
        if len(expected) != len(got):
            return [f"{path}: length {len(expected)} -> {len(got)}"]
        for i, (e, g) in enumerate(zip(expected, got)):
            bad += diff(e, g, f"{path}[{i}]")
        return bad
    if expected != got:
        bad.append(f"{path}: {expected!r} -> {got!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="record baselines (pre-edit, ONCE) instead of comparing")
    args = ap.parse_args()

    t0 = time.time()
    checks = compute()
    dt = time.time() - t0

    if args.write:
        doc = {
            "meta": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "obs_lock_dims": OBS_LOCK,
                "steps": STEPS,
            },
            "checks": checks,
        }
        with open(BASELINE_PATH, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"baselines written -> {BASELINE_PATH}  ({dt:.1f}s)")
        for name, d in checks["drivetrain"].items():
            print(f"  drivetrain {name}: checksum={float.fromhex(d['checksum']):.12e}")
        for name, d in checks["tracks"].items():
            print(f"  track {name}: traj={float.fromhex(d['traj_checksum']):.12e}  "
                  f"offtrack={d['offtrack_steps']}  airborne={d['airborne_steps']}  "
                  f"obs_samples={d['obs_count']}")
        return 0

    if not os.path.exists(BASELINE_PATH):
        print(f"FAIL: no baseline file at {BASELINE_PATH} — run with --write first "
              f"(BEFORE any physics edit).")
        return 2
    with open(BASELINE_PATH) as f:
        doc = json.load(f)

    bad = diff(doc["checks"], checks)
    for t, d in checks["tracks"].items():
        if d["airborne_steps"] != 0:
            bad.append(f"/tracks/{t}/airborne_steps: flat-track takeoff! "
                       f"({d['airborne_steps']} steps airborne)")

    if bad:
        print(f"FAIL: {len(bad)} mismatch(es) vs pre-hills baseline ({dt:.1f}s):")
        for b in bad[:40]:
            print(f"  {b}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        return 1
    print(f"PASS: byte-identical to pre-hills baseline "
          f"({doc['meta']['created']}); airborne_steps=0  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
