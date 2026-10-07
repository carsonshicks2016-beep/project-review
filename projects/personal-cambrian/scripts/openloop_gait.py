#!/usr/bin/env python3
"""Drive a seed with a hand-coded open-loop CPG gait and report forward progress.

    python3 scripts/openloop_gait.py                 # both seeds, gait vs still
    python3 scripts/openloop_gait.py --render        # also save a walked-forward PNG

No learning involved (that's Stage 3) — this just shows the evaluation stack works
end to end: control input -> forward motion -> reward.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.sim.openloop import open_loop_rollout, zero_action_rollout, cpg, ramp_phases

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAITS = {"agent_zero": dict(freq=2.5, amp=1.0, phase="alt"),
         "quadruped": dict(freq=1.5, amp=1.0, phase="ramp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="save a PNG after walking")
    args = ap.parse_args()

    for name, fn in (("agent_zero", agent_zero), ("quadruped", quadruped)):
        env = CreatureEnv(fn(), task=LocomotionTask(max_steps=400))
        gait = open_loop_rollout(env, steps=400, seed=0, **GAITS[name])
        still = zero_action_rollout(CreatureEnv(fn(), task=LocomotionTask(max_steps=400)),
                                    steps=400, seed=0)
        print(f"{name:11s} gait net_x={gait['net_x']:+.3f}m (survived {gait['survived']}) "
              f"| still net_x={still['net_x']:+.3f}m | forward_reward={gait['forward_reward']:+.2f}")
        env.close()

    if args.render:
        from view_creature import write_png
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        env = CreatureEnv(quadruped(), task=LocomotionTask(max_steps=400),
                          render_mode="rgb_array")
        env.reset(seed=0)
        phases = ramp_phases(env.action_space.shape[0])
        t = 0.0
        for _ in range(120):
            env.step(cpg(t, phases, 1.5, 1.0).astype(np.float32))
            t += env.control_dt
        out = os.path.join(ROOT, "renders", "quadruped_walked.png")
        write_png(out, env.render())
        print(f"rendered -> {out}")
        env.close()


if __name__ == "__main__":
    main()
