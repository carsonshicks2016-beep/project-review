"""Drivetrain smoke/regression helper.

Supra/RX-7 exercise the legacy RWD+clutch fast path; Skyline exercises the
R34 GT-R ATTESA-style AWD path.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supra.config import get_car, SimSpec
from supra.physics import Vehicle, Controls

def run_car(name):
    v = Vehicle(get_car(name), SimSpec()); v.reset(0, 0, 0, speed=0.0)
    rec = []
    for i in range(3000):
        thr = 0.8 if i % 600 < 400 else 0.0
        brk = 0.6 if i % 600 >= 500 else 0.0
        steer = 0.5 * np.sin(i * 0.01)
        v.step(Controls(steer=steer, throttle=thr, brake=brk,
                        clutch=1.0 if thr > 0 else 0.0))
        rec.append((v.x, v.y, v.yaw, v.vx, v.vy, v.r, v.engine_w, *v.wheel_w))
    return np.array(rec)

if __name__ == "__main__":
    for name in ("supra", "rx7", "skyline"):
        a = run_car(name)
        print(f"{name}: checksum={np.sum(a):.12e}  final_xy=({a[-1,0]:.6f},{a[-1,1]:.6f})  v={np.hypot(a[-1,3],a[-1,4])*3.6:.3f}km/h")
