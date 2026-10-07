#!/usr/bin/env python3
"""Quantify warm-start transfer speedup across morphologies (ROADMAP Stage 3.6).

For each small-mutation child, measure the knee (steps to reach 90% of its own
plateau) when fine-tuning warm-started-from-parent vs from-scratch. The per-child
speedup is scratch_knee / warm_knee; we report (and log) the median.

    python3 scripts/transfer_test.py --n-bodies 6
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.encoding import Genome
from personal_cambrian.seeds import quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import (
    PPOConfig, make_modular_setup, warmstart_curve, budget_curve, scratch_make_agent,
    find_knee,
)
from personal_cambrian.control.ppo import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def small_mutation(g, rng, sigma=0.12):
    child = Genome.from_json(g.to_json())
    for p in child.parts:
        p.dims = {k: max(0.01, v * (1 + rng.normal(0, sigma))) for k, v in p.dims.items()}
    for m in child.muscles:
        m.pcsa_cm2 = max(1.0, m.pcsa_cm2 * (1 + rng.normal(0, sigma)))
    return child


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-steps", type=int, default=200_000)
    ap.add_argument("--warm-steps", type=int, default=16_000)
    ap.add_argument("--scratch-steps", type=int, default=100_000)
    ap.add_argument("--n-bodies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pmake_env, pmake_agent = make_modular_setup("quadruped")
    print(f"training parent ({args.parent_steps} steps)...")
    parent, _ = train(pmake_env, PPOConfig(total_timesteps=args.parent_steps, seed=args.seed),
                      make_agent=pmake_agent)

    rows, speedups = [], []
    print(f"\nmeasuring transfer speedup over {args.n_bodies} morphologies:")
    for i in range(args.n_bodies):
        g = quadruped() if i == 0 else small_mutation(quadruped(), rng)
        cme = (lambda gg: (lambda s: CreatureEnv(gg, task=LocomotionTask(max_steps=500),
                                                 obs_mode="structured")))(g)
        wk = find_knee(*warmstart_curve(cme, parent, max_steps=args.warm_steps))
        sk = find_knee(*budget_curve(cme, scratch_make_agent(cme), max_steps=args.scratch_steps))
        sp = sk / max(wk, 1)
        speedups.append(sp)
        rows.append({"body": i, "warm_knee": wk, "scratch_knee": sk, "speedup": sp})
        print(f"  body {i}: warm_knee={wk:6d}  scratch_knee={sk:6d}  speedup={sp:5.1f}x")

    median = float(np.median(speedups))
    out = os.path.join(ROOT, "runs", f"transfer-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump({"median_speedup": median, "per_body": rows}, f, indent=2)
    print(f"\nMEDIAN warm-start transfer speedup = {median:.1f}x  "
          f"(target >=3x: {'PASS' if median >= 3 else 'below target'})")
    print(f"logged -> {out}")


if __name__ == "__main__":
    main()
