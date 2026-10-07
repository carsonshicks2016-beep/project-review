"""Airborne physics validation — Gate 3 of PHYSICS_3D_PLAN.md.

Exercises takeoff / ballistic flight / landing on ANALYTIC synthetic roads
fed straight into Vehicle.set_road() (track call-site wiring is Stage 4):

  * takeoff threshold — over a Gaussian crest with launch speed v* =
    sqrt(g*w^2/2h), crossing at 0.98*v* stays planted (loads dip + recover),
    1.02*v* launches within metres of the crest top.
  * ramp launch      — off an 18% ramp with a rounded lip, launch vz matches
    vx*grade at the lip within 3%.
  * ballistic flight — yaw rate is EXACTLY constant (zero tyre moment),
    energy (KE + m*g*z) balances against drag work within 2%, and the
    landing point matches the drag-free parabola within 8%.
  * free-rev         — full throttle in the air spins the engine up
    (no load on the driven wheels).
  * landing          — touchdown re-grounds the car, landing_g matches the
    absorbed vertical speed, the impact spike decays and loads settle back
    to static within ~6 suspension_tau; never NaN, never negative.

Usage: PYTHONPATH="$PWD" python3 tools/validate_jumps.py
Exits non-zero on any failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supra.config import SimSpec, get_car
from supra.physics import G, Controls, Vehicle

FAILURES = []
SPEC = get_car("supra")
DT = SimSpec().dt


def check(name: str, ok: bool, detail: str = ""):
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def fresh(speed=0.0) -> Vehicle:
    v = Vehicle(SPEC, SimSpec())
    v.reset(0.0, 0.0, 0.0, speed=speed)
    return v


class StraightRoad:
    """Analytic road along +x (heading 0): grade(s)/vcurv(s) closed-form,
    z(s) integrated once on a fine grid. Feeds set_road from the car's x."""

    def __init__(self, grade_fn, vcurv_fn, s_max=600.0):
        self.grade_fn, self.vcurv_fn = grade_fn, vcurv_fn
        self._s = np.arange(0.0, s_max, 0.01)
        self._z = np.cumsum([grade_fn(s) * 0.01 for s in self._s])

    def z(self, s: float) -> float:
        return float(np.interp(s, self._s, self._z))

    def feed(self, veh: Vehicle):
        s = veh.x
        veh.set_road(self.grade_fn(s), 0.0, 0.0, self.z(s), self.vcurv_fn(s))


# --------------------------------------------------------------------------- #
# Gaussian crest: z = h*exp(-(d/w)^2), launch speed v* = sqrt(g*w^2 / 2h)
# --------------------------------------------------------------------------- #
H, W, S0 = 2.0, 20.0, 150.0
V_LAUNCH = float(np.sqrt(G * W * W / (2.0 * H)))


def crest_grade(s):
    d = s - S0
    return float(-2.0 * d * H * np.exp(-(d / W) ** 2) / (W * W))


def crest_vcurv(s):
    d = s - S0
    return float((4.0 * d * d / W ** 4 - 2.0 / W ** 2) * H * np.exp(-(d / W) ** 2))


def run_crest(v_pin: float):
    """Cross the crest with vx pinned (isolates the takeoff threshold)."""
    road = StraightRoad(crest_grade, crest_vcurv)
    v = fresh(speed=v_pin)
    min_fz, x_takeoff = 1e18, None
    for _ in range(int(12.0 / DT)):
        v.vx = v_pin
        road.feed(v)
        v.step(Controls(throttle=0.3, clutch=1.0))
        min_fz = min(min_fz, float(v.Fz.sum()))
        if v.airborne and x_takeoff is None:
            x_takeoff = v.x
            break
        if v.x > S0 + 90.0:
            break
    return x_takeoff, min_fz, v


def test_takeoff_threshold():
    print(f"takeoff threshold (Gaussian crest, v* = {V_LAUNCH:.2f} m/s):")
    x_off, min_fz, v = run_crest(0.98 * V_LAUNCH)
    static = SPEC.mass * G
    check("0.98*v*: stays planted, loads dip and recover",
          x_off is None and min_fz < 0.3 * static
          and float(v.Fz.sum()) > 0.85 * static,
          f"min {min_fz/static:.2f}x static, final {v.Fz.sum()/static:.2f}x")
    x_off, _, _ = run_crest(1.02 * V_LAUNCH)
    check("1.02*v*: launches at the crest top",
          x_off is not None and abs(x_off - S0) < 3.0,
          f"takeoff at x={x_off if x_off else 'never'}" if x_off is None
          else f"takeoff at x={x_off:.1f} (crest {S0:.0f})")


# --------------------------------------------------------------------------- #
# ramp + tabletop: 18% ramp, 2.5 m rounded lip, flat plateau after
# --------------------------------------------------------------------------- #
S1, S2, S3, LLIP, G0 = 60.0, 70.0, 130.0, 2.5, 0.18


def ramp_grade(s):
    if s < S1 or s >= S3 + LLIP:
        return 0.0
    if s < S2:
        return G0 * (1.0 - np.cos(np.pi * (s - S1) / (S2 - S1))) / 2.0
    if s < S3:
        return G0
    return G0 * (1.0 + np.cos(np.pi * (s - S3) / LLIP)) / 2.0


def ramp_vcurv(s):
    if S1 <= s < S2:
        return G0 * np.pi * np.sin(np.pi * (s - S1) / (S2 - S1)) / (2.0 * (S2 - S1))
    if S3 <= s < S3 + LLIP:
        return -G0 * np.pi * np.sin(np.pi * (s - S3) / LLIP) / (2.0 * LLIP)
    return 0.0


def test_ramp_flight():
    print("ramp launch + ballistic flight + landing (18% ramp, 25 m/s):")
    s = SPEC
    road = StraightRoad(ramp_grade, ramp_vcurv)
    v = fresh(speed=25.0)
    take = land = None                       # (x, vx, vz, z, r, rpm, t)
    r_drift, e_resid_max, w_drag = 0.0, 0.0, 0.0
    fz_bad, t = False, 0.0
    settle = []
    for _ in range(int(25.0 / DT)):
        road.feed(v)
        thr = 1.0 if v.airborne else float(np.clip(0.4 + 1.2 * (25.0 - v.speed), 0.0, 0.9))
        v.step(Controls(steer=0.0, throttle=thr, clutch=1.0))
        t += DT
        fz_bad |= bool(np.any(~np.isfinite(v.Fz)) or np.any(v.Fz < 0.0))
        if v.airborne:
            if take is None:
                take = (v.x, v.vx, v.vz, v.z, v.r, v.rpm, t)
                e0 = 0.5 * s.mass * (v.vx**2 + v.vy**2 + v.vz**2) + s.mass * G * v.z
            r_drift = max(r_drift, abs(v.r - take[4]))
            drag = (0.5 * s.air_density * s.drag_area * v.vx * abs(v.vx) * v.vx
                    + 0.5 * s.air_density * s.side_drag_area * v.vy * abs(v.vy) * v.vy)
            w_drag += abs(drag) * DT
            e = 0.5 * s.mass * (v.vx**2 + v.vy**2 + v.vz**2) + s.mass * G * v.z
            e_resid_max = max(e_resid_max, abs(e - e0 + w_drag))
        elif take is not None and land is None:
            land = (v.x, v.vx, v.vz, v.z, v.r, v.rpm, t)
            vz_before = prev_vz
            t_land = t
        if take is not None and land is not None and t - t_land > 1.2:
            settle.append(float(v.Fz.sum()))
            if len(settle) > 30:
                break
        prev_vz = v.vz
    if take is None or land is None:
        check("car launches off the lip and lands", False,
              f"take={take is not None} land={land is not None}")
        return

    x0, vx0, vz0, z0, r0, rpm0, t0 = take
    check("launch vz = vx*grade at the lip (<=3%)",
          abs(vz0 - vx0 * G0) <= 0.03 * vx0 * G0,
          f"vz {vz0:.2f} vs vx*0.18 = {vx0*G0:.2f} "
          f"(launched at x={x0:.1f}, lip {S3:.0f}-{S3+LLIP:.1f})")
    check("yaw rate EXACTLY constant in flight (zero tyre moment)",
          r_drift == 0.0, f"max drift {r_drift:.2e} (r0 {r0:+.3f})")
    e_ref = 0.5 * s.mass * vx0 * vx0
    check("flight energy audit: d(KE + mgz) + drag work <= 2%",
          e_resid_max <= 0.02 * e_ref,
          f"max residual {e_resid_max/1e3:.2f}kJ of {e_ref/1e3:.0f}kJ "
          f"({100*e_resid_max/e_ref:.2f}%), drag work {w_drag/1e3:.1f}kJ")

    # drag-free parabola from the takeoff state, landing at plateau height
    z_top = road.z(S3 + LLIP + 1.0)
    tf = (vz0 + np.sqrt(max(vz0 * vz0 - 2.0 * G * (z_top - z0), 0.0))) / G
    r_pred = vx0 * tf
    r_sim = land[0] - x0
    check("flight range matches the parabola (<=8%)",
          abs(r_sim - r_pred) <= 0.08 * r_pred,
          f"flew {r_sim:.1f}m vs predicted {r_pred:.1f}m "
          f"({land[6]-t0:.2f}s air, {100*abs(r_sim-r_pred)/r_pred:.1f}% off)")
    check("full throttle in the air free-revs the engine",
          land[5] > rpm0 + 500.0, f"rpm {rpm0:.0f} -> {land[5]:.0f}")

    tau = SPEC.landing_tau
    pred_g = (0.0 - vz_before) / (G * tau)
    check("landing_g matches the absorbed vertical speed (<=10%)",
          abs(v.landing_g - pred_g) <= 0.10 * abs(pred_g) + 0.05,
          f"landing_g {v.landing_g:.2f} vs predicted {pred_g:.2f} "
          f"(touched down at {abs(vz_before):.1f} m/s)")
    df = 0.5 * s.air_density * s.downforce_ClA * v.vx * v.vx
    static = s.mass * G + df
    settled = np.mean(settle[-10:])
    check("loads settle to static within ~6 suspension_tau",
          abs(settled - static) <= 0.03 * static,
          f"settled {settled/static:.3f}x static")
    check("Fz finite and non-negative throughout", not fz_bad)
    check("re-grounded on the plateau",
          not v.airborne and abs(v.z - z_top) < 1e-6,
          f"z {v.z:.2f} vs plateau {z_top:.2f}")


def test_spin_persistence():
    """A spin thrown before takeoff CONTINUES in the air (Mz = 0 exactly)."""
    print("spin persistence over the crest (1.08*v*, r injected at takeoff):")
    road = StraightRoad(crest_grade, crest_vcurv)
    v = fresh(speed=1.08 * V_LAUNCH)
    target = 1.08 * V_LAUNCH
    injected = False
    drift, air_steps = 0.0, 0
    for _ in range(int(15.0 / DT)):
        road.feed(v)
        thr = 0.0 if v.airborne else float(np.clip(0.4 + 1.2 * (target - v.speed), 0.0, 0.9))
        v.step(Controls(throttle=thr, clutch=1.0))
        if v.airborne:
            if not injected:
                v.r = 0.4                     # the drift you brought to the lip
                injected = True
            else:
                drift = max(drift, abs(v.r - 0.4))
                air_steps += 1
        elif injected:
            break
    check("took off, spun through the air, landed",
          injected and air_steps > 20 and not v.airborne,
          f"{air_steps} airborne steps")
    check("yaw rate frozen mid-air (bit-exact)", drift == 0.0,
          f"max drift {drift:.2e}")


if __name__ == "__main__":
    for fn in (test_takeoff_threshold, test_ramp_flight,
               test_spin_persistence):
        fn()
    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("\nPASS: airborne physics validated (Gate 3)")
