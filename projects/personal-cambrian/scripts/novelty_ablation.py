#!/usr/bin/env python3
"""Novelty-search ablation on the real creature pipeline (ROADMAP Stage 8.2).

Runs the Stage-5 QD loop twice from the same seed and budget -- once with the NSLC
novelty emitter ON, once OFF (plain MAP-Elites sampling) -- and compares how much
of *behavior* and *descriptor* space each explores. The Stage-8.2 done-when is that
the novelty-on run explores MORE:

    python3 scripts/novelty_ablation.py
    python3 scripts/novelty_ablation.py --iterations 120 --fine-tune 3000

The fast, RL-free proof of the same mechanism is in tests/test_novelty.py; this is
the (slow) demonstration on actual trained creatures.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.evo import run_qd, QDConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(seed_genome, *, novelty, args):
    cfg = QDConfig(
        axes=["aspect", "limb_count"], bins=args.bins, iterations=args.iterations,
        seed_steps=args.seed_steps, fine_tune_steps=args.fine_tune,
        n_envs=4, n_steps=256, ep_steps=args.ep_steps, hidden=64,
        macro_rate=0.4, novelty=novelty, novelty_weight=1.0,
        local_competition_weight=1.0, track_behavior=True,   # both runs measure BC coverage
        behavior_radius=args.radius, seed=args.seed)
    logs = []
    t0 = time.time()
    archive, _ = run_qd([seed_genome], cfg, log_fn=logs.append)
    return archive, logs, time.time() - t0


def _summary(archive, logs):
    last = logs[-1] if logs else {}
    return {
        "coverage": archive.coverage,
        "cells": len(archive.grid),
        "qd_score": archive.qd_score,
        "bc_cells": last.get("bc_cells", float("nan")),
        "bc_spread": last.get("bc_spread", float("nan")),
        "bc_archive": last.get("bc_archive", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--seed-steps", type=int, default=20_000)
    ap.add_argument("--fine-tune", type=int, default=2_000)
    ap.add_argument("--ep-steps", type=int, default=200)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    genome = SEEDS[args.seed_creature]()
    print(f"novelty ablation: {args.seed_creature}, {args.iterations} iters, "
          f"fine-tune={args.fine_tune} steps")

    off_arch, off_logs, off_t = _run(genome, novelty=False, args=args)
    print(f"  novelty OFF done in {off_t:.0f}s")
    on_arch, on_logs, on_t = _run(genome, novelty=True, args=args)
    print(f"  novelty ON  done in {on_t:.0f}s")

    off, on = _summary(off_arch, off_logs), _summary(on_arch, on_logs)
    print(f"\n{'metric':<16}{'novelty OFF':>14}{'novelty ON':>14}")
    for key, label in [("coverage", "descriptor cov"), ("cells", "descriptor cells"),
                       ("qd_score", "qd score"), ("bc_cells", "behavior cells"),
                       ("bc_spread", "behavior spread"), ("bc_archive", "behaviors seen")]:
        fa, fb = off[key], on[key]
        fmt = (lambda v: f"{v:.0%}") if key == "coverage" else (
            (lambda v: f"{v:.3f}") if isinstance(fb, float) else (lambda v: f"{v}"))
        print(f"{label:<16}{fmt(fa):>14}{fmt(fb):>14}")

    # BC reach is judged by elite-count-independent measures: behavioral niches
    # discovered across all evals, and how spread out the survivors are.
    more_cells = on["bc_cells"] > off["bc_cells"]
    more_spread = on["bc_spread"] > off["bc_spread"]
    print(f"\nbehavioral niches (all evals): {off['bc_cells']} -> {on['bc_cells']}  ({'+' if more_cells else 'no +'})")
    print(f"survivor behavior spread:      {off['bc_spread']:.3f} -> {on['bc_spread']:.3f}  ({'+' if more_spread else 'no +'})")
    print(f"descriptor cells filled:       {off['cells']} -> {on['cells']}")
    print("DONE-WHEN MET: novelty explores more behavior space"
          if (more_cells or more_spread) else "no behavior-space gain at this budget")


if __name__ == "__main__":
    main()
