"""2.5D slope physics validation — Gate 2 of PHYSICS_3D_PLAN.md.

Exercises the Stage 2 grounded physics by feeding Vehicle.set_road() directly
with synthetic road data (the track call-site wiring is Stage 4):

  * zero-road identity — calling set_road(0,0,yaw,0,0) every step is
    bit-identical to never calling it (the caller contract).
  * slope pull        — instantaneous ax difference on a 10% grade is -g*0.10.
  * energy audit      — coast-down on a constant climb: dKE + m*g*dh + losses
                        balance within 5%.
  * climb vs downhill — same throttle: climbing slows, descending runs away.
  * static loads      — settled loads on a 10% grade match the analytic
                        formula (rear axle gains m*g*grade*h/L, total scales
                        by cos-factor) to 1e-6.
  * crest/dip loads   — total load scales by 1 + vcurv*v^2/g, hits exactly
                        zero at the launch speed, clamps at 1.7 in dips,
                        never goes negative.
  * bank pull         — left-edge-up pulls the car toward the low (right)
                        side and loads the right wheels.
  * body projection   — a car sideways on a climb feels the slope as BANK
                        (under its side), not grade; pitch/roll outputs match.
  * banked corner     — physics-level: banked into the turn sustains higher
                        centripetal acceleration than flat.

Usage: PYTHONPATH="$PWD" python3 tools/validate_hills.py
Exits non-zero on any failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supra.config import SimSpec, get_car
from supra.physics import FL, FR, RL, RR, G, Controls, Vehicle

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


# --------------------------------------------------------------------------- #
def test_zero_road_identity():
    print("zero-road identity (the byte-identity contract, unit level):")
    a, b = fresh(), fresh()
    rec_a, rec_b = [], []
    for i in range(1500):
        thr = 0.8 if i % 600 < 400 else 0.0
        brk = 0.6 if i % 600 >= 500 else 0.0
        steer = 0.5 * np.sin(i * 0.01)
        c = Controls(steer=steer, throttle=thr, brake=brk,
                     clutch=1.0 if thr > 0 else 0.0)
        a.step(c)
        b.set_road(0.0, 0.0, b.yaw, 0.0, 0.0)
        b.step(c)
        rec_a.append((a.x, a.y, a.yaw, a.vx, a.vy, a.r, a.engine_w, *a.wheel_w))
        rec_b.append((b.x, b.y, b.yaw, b.vx, b.vy, b.r, b.engine_w, *b.wheel_w))
    check("set_road zeros == never called (bit-exact)",
          np.array_equal(np.array(rec_a), np.array(rec_b)))
    check("never airborne, vert factor pinned at 1.0",
          not b.airborne and b._vert_raw == 1.0)


def test_slope_pull():
    print("slope pull (instantaneous force):")
    ax = {}
    for grade in (0.0, 0.10):
        v = fresh(speed=20.0)
        v.set_road(grade, 0.0, v.yaw, 0.0)
        v.step(Controls(throttle=0.0, clutch=0.0))
        ax[grade] = v.ax
    d = ax[0.10] - ax[0.0]
    check("ax(10% climb) - ax(flat) = -g*0.10",
          abs(d + G * 0.10) < 0.02 * G, f"got {d:+.3f}, want {-G*0.10:+.3f}")


def test_energy_audit():
    print("energy audit (coast-down on a constant 8% climb):")
    s = SPEC
    grade, v0 = 0.08, 32.0
    v = fresh(speed=v0)
    dh, w_loss = 0.0, 0.0
    for _ in range(1200):                       # 10 s coast
        v.set_road(grade, 0.0, v.yaw, 0.0)
        v.step(Controls(throttle=0.0, clutch=0.0))
        spd = v.speed
        drag = (0.5 * s.air_density * s.drag_area * v.vx * abs(v.vx) * v.vx
                + 0.5 * s.air_density * s.side_drag_area * v.vy * abs(v.vy) * v.vy)
        rr = s.rolling_resistance * float(v.Fz.sum()) * min(1.0, spd / 0.5) * spd
        w_loss += (abs(drag) + rr) * DT
        dh += v.vx * grade * DT
    ke0 = 0.5 * s.mass * v0 * v0
    ke1 = 0.5 * s.mass * (v.vx * v.vx + v.vy * v.vy)
    resid = (ke1 - ke0) + s.mass * G * dh + w_loss
    check("dKE + m*g*dh + losses ~ 0 (<=5% of E0)",
          abs(resid) <= 0.05 * ke0,
          f"residual {resid/1e3:.1f}kJ of E0 {ke0/1e3:.1f}kJ "
          f"({100*abs(resid)/ke0:.1f}%), climbed {dh:.1f}m, "
          f"v {v0:.0f}->{v.speed:.1f}m/s")


def test_climb_vs_downhill():
    print("same throttle, three grades (8 s, fixed 3rd gear):")
    final = {}
    for grade in (0.08, 0.0, -0.08):
        v = fresh(speed=22.0)
        v.gear = 3
        for _ in range(960):
            v.set_road(grade, 0.0, v.yaw, 0.0)
            v.step(Controls(throttle=0.45, clutch=1.0))
        final[grade] = v.speed
    check("climb slows < flat < downhill runs away",
          final[0.08] + 1.5 < final[0.0] < final[-0.08] - 1.5,
          f"climb {final[0.08]:.1f} / flat {final[0.0]:.1f} / "
          f"down {final[-0.08]:.1f} m/s")


def settle_loads(v: Vehicle, n=400) -> np.ndarray:
    for _ in range(n):
        v._update_loads(0.05)
    return v.Fz


def test_static_loads():
    print("static loads on a 10% grade (analytic match):")
    s = SPEC
    v = fresh()
    v.grade_body = 0.10                  # unit-level: isolate _update_loads
    fz = settle_loads(v)
    cosn = 1.0 / np.sqrt(1.0 + 0.10 * 0.10)
    gt = s.mass * G * 0.10 * s.cg_height / s.wheelbase
    wf = s.mass * G * s.front_weight / 2.0
    wr = s.mass * G * (1.0 - s.front_weight) / 2.0
    want = np.array([wf * cosn - gt / 2.0, wf * cosn - gt / 2.0,
                     wr * cosn + gt / 2.0, wr * cosn + gt / 2.0])
    check("per-wheel loads match formula to 1e-6",
          bool(np.allclose(fz, want, rtol=1e-6)),
          f"rear axle gain {fz[RL]+fz[RR]-2*wr:+.0f}N (want {2*(wr*(cosn-1))+gt:+.0f}N)")
    check("rear axle gains, total scales by cos-factor",
          fz[RL] + fz[RR] > 2 * wr and abs(fz.sum() - s.mass * G * cosn) < 1.0)


def test_crest_dip_loads():
    print("v^2 crest/dip load factor:")
    s = SPEC
    vc = 0.008
    for vx, label in ((25.0, "below launch"),):
        v = fresh()
        v.vx = vx
        v.road_vcurv = -vc
        fz = settle_loads(v)
        df = 0.5 * s.air_density * s.downforce_ClA * vx * vx
        vert = 1.0 - vc * vx * vx / G
        want = (s.mass * G + df) * vert
        check(f"crest {label}: total = (mg+df)*(1+vcurv*v^2/g)",
              abs(fz.sum() - want) < 1e-3 * want,
              f"sum {fz.sum():.0f}N want {want:.0f}N (factor {vert:.3f})")
    v = fresh()
    v.vx = float(np.sqrt(G / vc))                 # exactly the launch speed
    v.road_vcurv = -vc
    fz = settle_loads(v)
    check("AT launch speed: total load -> 0, takeoff trigger visible",
          fz.sum() < 1.0 and v._vert_raw <= 1e-9,
          f"sum {fz.sum():.2e}N  vert_raw {v._vert_raw:.2e}")
    v = fresh()
    v.vx = 45.0                                   # beyond launch speed
    v.road_vcurv = -vc
    fz = settle_loads(v)
    check("beyond launch: clamped at 0, never negative",
          fz.sum() < 1.0 and v._vert_raw < 0.0 and np.all(fz >= 0.0),
          f"sum {fz.sum():.2e}N  vert_raw {v._vert_raw:+.3f}")
    v = fresh()
    v.vx = 40.0
    v.road_vcurv = +vc                            # dip: heavy, capped
    fz = settle_loads(v)
    df = 0.5 * s.air_density * s.downforce_ClA * 40.0 * 40.0
    want = (s.mass * G + df) * 1.7
    check("dip clamps at 1.7x", abs(fz.sum() - want) < 1e-3 * want,
          f"sum {fz.sum():.0f}N want {want:.0f}N")


def test_bank_pull():
    print("banking (left edge up -> pull + load toward the right):")
    v = fresh(speed=20.0)
    v.set_road(0.0, 0.08, v.yaw, 0.0)
    v.step(Controls(throttle=0.0, clutch=0.0))
    check("ay pulled toward the low side", v.ay < -0.5 * G * 0.08,
          f"ay {v.ay:+.2f} (pure gravity would be {-G*0.08:+.2f})")
    v2 = fresh()
    v2.bank_body = 0.08
    fz = settle_loads(v2)
    check("right wheels gain load",
          fz[FR] + fz[RR] > fz[FL] + fz[RL] + 100.0,
          f"right-left {fz[FR]+fz[RR]-fz[FL]-fz[RL]:+.0f}N")


def test_body_projection():
    print("slope projection onto body axes:")
    v = fresh()
    v.yaw = 1.0
    v.set_road(0.10, 0.0, 1.0, 0.0)               # driving along the tangent
    ok_along = (abs(v.grade_body - 0.10) < 1e-12 and abs(v.bank_body) < 1e-12
                and abs(v.pitch - np.arctan(0.10)) < 1e-12)
    v.set_road(0.10, 0.0, 1.0 - np.pi / 2.0, 0.0)  # car sideways on the climb
    ok_side = (abs(v.grade_body) < 1e-12 and abs(v.bank_body - 0.10) < 1e-12
               and abs(v.roll - np.arctan(0.10)) < 1e-12)
    check("along the road: slope under the nose, pitch output", ok_along)
    check("sideways on the climb: slope under the SIDE, roll output", ok_side)


def test_banked_corner():
    print("banked corner sustains more lateral (physics-level):")

    def max_centripetal(bank):
        v = fresh(speed=15.0)
        v.gear = 3
        best = 0.0
        for i in range(int(40.0 / DT)):
            v.set_road(0.0, bank, v.yaw, 0.0)     # carousel: bank tracks the car
            thr = min(0.85, 0.25 + i * DT * 0.02)
            v.step(Controls(steer=0.25, throttle=thr, clutch=1.0))
            if abs(v.slip_angle) < np.radians(20.0):
                best = max(best, abs(v.vx * v.r))
        return best

    flat = max_centripetal(0.0)
    banked = max_centripetal(-0.10)               # left turn, left edge low
    check("banked-in beats flat", banked > flat + 0.3,
          f"banked {banked:.2f} vs flat {flat:.2f} m/s^2")


if __name__ == "__main__":
    for fn in (test_zero_road_identity, test_slope_pull, test_energy_audit,
               test_climb_vs_downhill, test_static_loads, test_crest_dip_loads,
               test_bank_pull, test_body_projection, test_banked_corner):
        fn()
    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("\nPASS: 2.5D slope physics validated (Gate 2)")
